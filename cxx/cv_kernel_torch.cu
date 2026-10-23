#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void cost_volume_kernel(
    const __half* __restrict__ feat_ref, const __half* __restrict__ feat_src,
    const float* __restrict__ disp, const float* __restrict__ pose,
    __half* __restrict__ out,
    int vb, int C, int D, int H, int W, float inv_sqrt_c) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int hw = H * W;
  long total = (long)vb * D * hw;
  if (idx >= total) return;
  int j = idx % W, i = (idx / W) % H, d = (idx / hw) % D, b = idx / (hw * D);
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
  float nrm = rsqrtf(fmaxf(px*px + py*py + pz*pz, 1e-6f));
  px *= nrm; py *= nrm; pz *= nrm;
  float phi2 = atan2f(px, pz);
  float theta2 = asinf(fminf(fmaxf(py, -1.f), 1.f));
  float gx = ((phi2 + CUDART_PI_F) * (0.5f / CUDART_PI_F)) * 2.f - 1.f;
  float gy = ((theta2 + CUDART_PI_F * 0.5f) / CUDART_PI_F) * 2.f - 1.f;
  // align_corners=True + zeros padding (matches the original grid_sample call)
  float fx = (gx + 1.f) * 0.5f * (W - 1.f);
  float fy = (gy + 1.f) * 0.5f * (H - 1.f);
  if (fx < 0.f || fx > W - 1.f || fy < 0.f || fy > H - 1.f) {
    out[idx] = __float2half(0.f);   // zeros padding: out-of-bounds reads as 0
    return;
  }
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
    acc += s * __half2float(feat_ref[base_ref]);
    base_src += (long)H * W;
    base_ref += (long)H * W;
  }
  out[idx] = __float2half(acc * inv_sqrt_c);
}

torch::Tensor cost_volume(torch::Tensor feat_ref, torch::Tensor feat_src,
                          torch::Tensor disp, torch::Tensor pose) {
  int vb = feat_ref.size(0), C = feat_ref.size(1);
  int H = feat_ref.size(2), W = feat_ref.size(3);
  int D = disp.size(1);
  auto out = torch::empty({vb, D, H, W}, feat_src.options().dtype(torch::kHalf));
  int threads = 256;
  long total = (long)vb * D * H * W;
  int blocks = (int)((total + threads - 1) / threads);
  cost_volume_kernel<<<blocks, threads>>>(
      (__half*)feat_ref.data_ptr<at::Half>(), (__half*)feat_src.data_ptr<at::Half>(),
      disp.data_ptr<float>(), pose.data_ptr<float>(),
      (__half*)out.data_ptr<at::Half>(), vb, C, D, H, W,
      1.f / sqrtf((float)C));
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cost_volume", &cost_volume, "fused cost volume correlation (fp16)");
}
