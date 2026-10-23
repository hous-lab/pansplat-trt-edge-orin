// Fused cost-volume correlation kernel (equivalent to the composite of
// warp_with_pose_depth_candidates + correlation). Thread map: one thread per
// (vb, d, i, j) output element. Math mirrors warp_with_pose_depth_candidates:
//   dir_{i,j} = ERP direction; p = R*dir*depth + t; p /= max(|p|, 1e-3)
//   (phi,theta) = (atan2(p.x,p.z), asin(p.y)); bilinear(feat_src @ (u,v))
//   corr[d,i,j] = (1/sqrt(C)) * <feat_src_bilinear, feat_ref[:,i,j]>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define SQRT_C 11.313708498984761f  // sqrt(128); C is passed at runtime, not this constant

__global__ void cost_volume_kernel(
    const __half* __restrict__ feat_ref,  // [vb, C, H, W]
    const __half* __restrict__ feat_src,  // [vb, C, H, W]
    const float*  __restrict__ disp,      // [vb, D, H, W](disparity)
    const float*  __restrict__ pose,      // [vb, 4, 4](c2w)
    __half* __restrict__ out,             // [vb, D, H, W]
    int vb, int C, int D, int H, int W, float inv_sqrt_c) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int hw = H * W;
  long total = (long)vb * D * hw;
  if (idx >= total) return;

  int j = idx % W;
  int i = (idx / W) % H;
  int d = (idx / hw) % D;
  int b = idx / (hw * D);

  // ERP direction (matches points_grid)
  float u = (j + 0.5f) / W, vv = (i + 0.5f) / H;
  float phi = u * 2.f * CUDART_PI_F - CUDART_PI_F;
  float theta = vv * CUDART_PI_F - CUDART_PI_F * 0.5f;
  float dx = cosf(theta) * sinf(phi), dy = sinf(theta), dz = cosf(theta) * cosf(phi);

  const float* P = pose + b * 16;
  float px = P[0]*dx + P[1]*dy + P[2]*dz;
  float py = P[4]*dx + P[5]*dy + P[6]*dz;
  float pz = P[8]*dx + P[9]*dy + P[10]*dz;
  float dep = disp[(b * D + d) * hw + i * W + j];
  px = px * dep + P[3]; py = py * dep + P[7]; pz = pz * dep + P[11];

  float nrm = rsqrtf(fmaxf(px*px + py*py + pz*pz, 1e-3f));
  px *= nrm; py *= nrm; pz *= nrm;
  float phi2 = atan2f(px, pz);
  float theta2 = asinf(fminf(fmaxf(py, -1.f), 1.f));
  float gx = ((phi2 + CUDART_PI_F) * (0.5f / CUDART_PI_F)) * 2.f - 1.f;
  float gy = ((theta2 + CUDART_PI_F * 0.5f) / CUDART_PI_F) * 2.f - 1.f;

  // bilinear 4-neighborhood
  float fx = (gx + 1.f) * 0.5f * W - 0.5f;
  float fy = (gy + 1.f) * 0.5f * H - 0.5f;
  fx = fminf(fmaxf(fx, 0.f), W - 1.f);
  fy = fminf(fmaxf(fy, 0.f), H - 1.f);
  int x0 = (int)fx, y0 = (int)fy;
  int x1 = min(x0 + 1, W - 1), y1 = min(y0 + 1, H - 1);
  float wx = fx - x0, wy = fy - y0;

  float acc = 0.f;
  long base_ref = ((long)b * C) * H * W + (long)i * W + j;
  long base_src = ((long)b * C) * H * W;
  for (int c = 0; c < C; ++c) {
    float s00 = __half2float(feat_src[base_src + (long)y0 * W + x0]);
    float s01 = __half2float(feat_src[base_src + (long)y0 * W + x1]);
    float s10 = __half2float(feat_src[base_src + (long)y1 * W + x0]);
    float s11 = __half2float(feat_src[base_src + (long)y1 * W + x1]);
    float s = s00*(1-wx)*(1-wy) + s01*wx*(1-wy) + s10*(1-wx)*wy + s11*wx*wy;
    acc += s * __half2float(feat_ref[base_ref + (long)c * H * W]);
    base_src += (long)H * W;
    base_ref += (long)H * W;
  }
  out[idx] = __float2half(acc * inv_sqrt_c);
}

struct Size { int vb, C, D, H, W; };

double bench(const Size& s, int iters) {
  long hw = (long)s.H * s.W;
  size_t feat = (size_t)s.vb * s.C * hw;
  size_t disp = (size_t)s.vb * s.D * hw;
  __half *fr, *fs, *o;
  float *dp, *po;
  cudaMalloc(&fr, feat * 2); cudaMalloc(&fs, feat * 2);
  cudaMalloc(&dp, disp * 4); cudaMalloc(&po, 16 * s.vb * 4);
  cudaMalloc(&o, disp * 2);
  cudaMemset(fr, 0x3c, feat * 2); cudaMemset(fs, 0x3c, feat * 2);
  cudaMemset(dp, 0, disp * 4); cudaMemset(po, 0, 16 * s.vb * 4);
  int threads = 256;
  long total = (long)s.vb * s.D * hw;
  int blocks = (int)((total + threads - 1) / threads);
  float inv_c = 1.f / sqrtf((float)s.C);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  // warmup
  for (int i = 0; i < 5; ++i)
    cost_volume_kernel<<<blocks, threads>>>(fr, fs, dp, po, o, s.vb, s.C, s.D, s.H, s.W, inv_c);
  cudaEventRecord(e0);
  for (int i = 0; i < iters; ++i)
    cost_volume_kernel<<<blocks, threads>>>(fr, fs, dp, po, o, s.vb, s.C, s.D, s.H, s.W, inv_c);
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float ms = 0; cudaEventElapsedTime(&ms, e0, e1); ms /= iters;
  cudaFree(fr); cudaFree(fs); cudaFree(dp); cudaFree(po); cudaFree(o);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return ms;
}

int main() {
  // true sizes of the three cost-volume stages (512 config)
  Size s0 = {2, 128, 128, 64, 128};
  Size s1 = {2, 96, 64, 128, 256};
  Size s2 = {2, 64, 32, 256, 512};
  double t0 = bench(s0, 20), t1 = bench(s1, 20), t2 = bench(s2, 20);
  printf("CV stage0: %.3f ms (TRT: 184.5)\n", t0);
  printf("CV stage1: %.3f ms (TRT: 120.1)\n", t1);
  printf("CV stage2: %.3f ms (TRT:  59.7)\n", t2);
  printf("CV total : %.3f ms (TRT: 364.3)  -> speedup %.1fx\n", t0+t1+t2, 364.3/(t0+t1+t2));
  return 0;
}
