// =====================================================================
// flash_attn.cu - forward flash attention (online softmax) for PanSplat
// cv_post1 / cv_post2 cross-view window attention.
//
//   shapes : H in {1,2}, S in {4096, 16384}, D = 32, fp16 in/out, fp32 accum
//   layout : Q,K,V,O contiguous [H][S][D] row-major
//   math   : O = softmax(Q @ K^T * scale) @ V,  scale = 1/sqrt(D), no mask
//   tiling : BR=64 query rows / CTA (4 warps x 16 rows), BC=64 keys / step
//   tensor cores : mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
//   The 16384x16384 fp16 score matrix never leaves registers.
//
//   sm_75+ ; compiled sm_89 (RTX 4090) and sm_86 (A6000 / Orin-class)
// =====================================================================
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define BR 64        // query rows per CTA
#ifndef BC
#define BC 64        // k/v columns per online-softmax step
#endif
#define DW 32        // head dim (fixed for cv_post1/cv_post2)
#define SS 40        // smem row stride in halves: 32 + 8 pad
#define NT (BC / 8)  // n-tiles (of 8) per step
#define NTO 4        // n-tiles in output       = DW/8

// ---- mma.m16n8k16: A 16x16, B 16x8, C 16x8 fp32. g=lane/4, c=lane%4 ----
__device__ __forceinline__ void mma_16816(const uint32_t a[4], const uint32_t b[2],
                                          float d[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// quad (lanes 4g..4g+3) owns C-fragment rows g and g+8 -> quad shuffle reduce
__device__ __forceinline__ float qred_max(float v) {
#pragma unroll
  for (int o = 1; o < 4; o <<= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float qred_sum(float v) {
#pragma unroll
  for (int o = 1; o < 4; o <<= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// ---- CTA tile loader: 128 threads, [64][32] halves gmem -> smem (stride SS)
// qscale != 0 : scale halves by qscale (used for Q, exp2-domain scaling)
__device__ __forceinline__ void load_tile(const __half* __restrict__ g, __half* sm,
                                          float qscale) {
  int row = threadIdx.x >> 2;       // 0..31
  int ch = (threadIdx.x & 3) << 3;  // 0,8,16,24
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int r = row + 32 * i;
    uint4 v = *reinterpret_cast<const uint4*>(g + (size_t)r * DW + ch);
    __half* dst = sm + (size_t)r * SS + ch;
    if (qscale != 0.f) {
      const __half* h = reinterpret_cast<const __half*>(&v);
      __half o[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) o[k] = __float2half(__half2float(h[k]) * qscale);
      *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<uint4*>(o);
    } else {
      *reinterpret_cast<uint4*>(dst) = v;
    }
  }
}

extern "C" __global__ void __launch_bounds__(128)
flash_fwd_kernel(const __half* __restrict__ Q, const __half* __restrict__ K,
                 const __half* __restrict__ V, __half* __restrict__ O, int H, int S,
                 float qscale /* = scale*log2(e) : exp2-domain online softmax */) {
  const int h = blockIdx.y;
  const int q0 = blockIdx.x * BR;          // first query row of this CTA
  const size_t base = (size_t)h * S * DW;
  const __half* Qp = Q + base + (size_t)q0 * DW;

  __shared__ __half sQ[BR * SS];
  __shared__ __half sK[BC * SS];
  __shared__ __half sV[BC * SS];

  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int c = lane & 3;
  const int warp = threadIdx.x >> 5;       // warp owns query rows [16w, 16w+16)

  load_tile(Qp, sQ, qscale);               // Q tile, pre-scaled
  __syncthreads();

  float oacc[NTO][4];                      // output accumulator (fp32)
#pragma unroll
  for (int t = 0; t < NTO; ++t)
#pragma unroll
    for (int r = 0; r < 4; ++r) oacc[t][r] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY;    // running row max, row g / g+8
  float l0 = 0.f, l1 = 0.f;                // running row sum, row g / g+8

  // A-fragment of Q: rows g/g+8 of this warp's 16-row band, cols 2c / 2c+8
  const uint32_t aQ[2][4] = {{
      *(const uint32_t*)(sQ + (size_t)(warp * 16 + g) * SS + 2 * c),
      *(const uint32_t*)(sQ + (size_t)(warp * 16 + g + 8) * SS + 2 * c),
      *(const uint32_t*)(sQ + (size_t)(warp * 16 + g) * SS + 2 * c + 8),
      *(const uint32_t*)(sQ + (size_t)(warp * 16 + g + 8) * SS + 2 * c + 8)},
      {*(const uint32_t*)(sQ + (size_t)(warp * 16 + g) * SS + 2 * c + 16),
       *(const uint32_t*)(sQ + (size_t)(warp * 16 + g + 8) * SS + 2 * c + 16),
       *(const uint32_t*)(sQ + (size_t)(warp * 16 + g) * SS + 2 * c + 24),
       *(const uint32_t*)(sQ + (size_t)(warp * 16 + g + 8) * SS + 2 * c + 24)}};

  // K/V tiles: gmem -> registers (prefetch) -> smem, so the gmem latency of
  // tile it+1 overlaps the mma/softmax work of tile it.
  const __half* Kp = K + base;
  const __half* Vp = V + base;
  uint4 kr[2], vr[2];
  const int lrow = threadIdx.x >> 2;      // 0..31
  const int lch = (threadIdx.x & 3) << 3; // 0,8,16,24
#pragma unroll
  for (int i = 0; i < 2; i++) {
    const int r = lrow + 32 * i;
    kr[i] = *(const uint4*)(Kp + (size_t)r * DW + lch);
    vr[i] = *(const uint4*)(Vp + (size_t)r * DW + lch);
  }

  for (int it = 0, nstep = S / BC; it < nstep; ++it) {
    // registers -> smem for this step's K/V tiles
#pragma unroll
    for (int i = 0; i < 2; i++) {
      const int r = lrow + 32 * i;
      *(uint4*)(sK + (size_t)r * SS + lch) = kr[i];
      *(uint4*)(sV + (size_t)r * SS + lch) = vr[i];
    }
    __syncthreads();

    // prefetch next K/V tile into registers (overlaps the compute below)
    if (it + 1 < nstep) {
#pragma unroll
      for (int i = 0; i < 2; i++) {
        const int r = lrow + 32 * i;
        kr[i] = *(const uint4*)(Kp + (size_t)((it + 1) * BC + r) * DW + lch);
        vr[i] = *(const uint4*)(Vp + (size_t)((it + 1) * BC + r) * DW + lch);
      }
    }

    // ---- S = Q @ K^T : warp tile 16 x 64, k = D = 32 (2 mma k-steps) ----
    float s[NT][4];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      s[t][0] = s[t][1] = s[t][2] = s[t][3] = 0.f;
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const uint32_t b[2] = {
            *(const uint32_t*)(sK + (size_t)(8 * t + g) * SS + 2 * c + 16 * kk),
            *(const uint32_t*)(sK + (size_t)(8 * t + g) * SS + 2 * c + 8 + 16 * kk)};
        mma_16816(aQ[kk], b, s[t]);
      }
    }

    // ---- online softmax, rows g (frag r=0,1) and g+8 (frag r=2,3) ----
    float rmax0 = -INFINITY, rmax1 = -INFINITY;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      rmax0 = fmaxf(rmax0, fmaxf(s[t][0], s[t][1]));
      rmax1 = fmaxf(rmax1, fmaxf(s[t][2], s[t][3]));
    }
    rmax0 = qred_max(rmax0);
    rmax1 = qred_max(rmax1);
    const float m_new0 = fmaxf(m0, rmax0);
    const float m_new1 = fmaxf(m1, rmax1);
    const float alpha0 = (m0 == -INFINITY) ? 0.f : exp2f(m0 - m_new0);
    const float alpha1 = (m1 == -INFINITY) ? 0.f : exp2f(m1 - m_new1);

    float rsum0 = 0.f, rsum1 = 0.f;
    uint32_t p[NT / 2][4];                 // P as A-fragments for PV mma
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      const float e0 = exp2f(s[t][0] - m_new0);
      const float e1 = exp2f(s[t][1] - m_new0);
      const float e2 = exp2f(s[t][2] - m_new1);
      const float e3 = exp2f(s[t][3] - m_new1);
      rsum0 += e0 + e1;
      rsum1 += e2 + e3;
      const __half2 h01 = __floats2half2_rn(e0, e1);
      const __half2 h23 = __floats2half2_rn(e2, e3);
      const int j = (t & 1) * 2;           // even n-tile -> a0/a1, odd -> a2/a3
      p[t >> 1][j + 0] = *reinterpret_cast<const uint32_t*>(&h01);
      p[t >> 1][j + 1] = *reinterpret_cast<const uint32_t*>(&h23);
    }
    rsum0 = qred_sum(rsum0);
    rsum1 = qred_sum(rsum1);

    const float l_new0 = l0 * alpha0 + rsum0;
    const float l_new1 = l1 * alpha1 + rsum1;
#pragma unroll
    for (int t = 0; t < NTO; ++t) {        // rescale running O to the new max
      oacc[t][0] *= alpha0;
      oacc[t][1] *= alpha0;
      oacc[t][2] *= alpha1;
      oacc[t][3] *= alpha1;
    }
    // ---- O = P @ V : warp tile 16 x 32, k = BC = 64 (4 mma k-steps) ----
    // B fragment of V: b0 = {V[16ks+2c][8t+g], V[16ks+2c+1][8t+g]}
#pragma unroll
    for (int t = 0; t < NTO; ++t) {
#pragma unroll
      for (int ks = 0; ks < NT / 2; ++ks) {
        const __half v0 = sV[(size_t)(16 * ks + 2 * c) * SS + 8 * t + g];
        const __half v1 = sV[(size_t)(16 * ks + 2 * c + 1) * SS + 8 * t + g];
        const __half v2 = sV[(size_t)(16 * ks + 2 * c + 8) * SS + 8 * t + g];
        const __half v3 = sV[(size_t)(16 * ks + 2 * c + 9) * SS + 8 * t + g];
        uint32_t b[2];
        *reinterpret_cast<__half2*>(&b[0]) = __halves2half2(v0, v1);
        *reinterpret_cast<__half2*>(&b[1]) = __halves2half2(v2, v3);
        mma_16816(p[ks], b, oacc[t]);
      }
    }

    // carry online-softmax state to the next step
    m0 = m_new0;
    m1 = m_new1;
    l0 = l_new0;
    l1 = l_new1;

    __syncthreads();   // all warps done reading sK/sV before next step overwrites
  }

  // ---- epilogue: O = oacc / l, store [16 rows x 32 cols] per warp ----
  const float inv_l0 = (l0 != 0.f) ? __frcp_rn(l0) : 1.f;
  const float inv_l1 = (l1 != 0.f) ? __frcp_rn(l1) : 1.f;
#pragma unroll
  for (int t = 0; t < NTO; ++t) {
    oacc[t][0] *= inv_l0;
    oacc[t][1] *= inv_l0;
    oacc[t][2] *= inv_l1;
    oacc[t][3] *= inv_l1;
  }
  const size_t orow = base + (size_t)(q0 + warp * 16) * DW;
#pragma unroll
  for (int t = 0; t < NTO; ++t) {
    *reinterpret_cast<__half2*>(O + orow + (size_t)g * DW + 8 * t + 2 * c) =
        __floats2half2_rn(oacc[t][0], oacc[t][1]);
    *reinterpret_cast<__half2*>(O + orow + (size_t)(g + 8) * DW + 8 * t + 2 * c) =
        __floats2half2_rn(oacc[t][2], oacc[t][3]);
  }
}

// ---------------- host launcher (C ABI, ctypes-friendly) ----------------
// S must be a multiple of BR=64; H in {1,2}; D fixed at 32.
extern "C" int flash_attn_fwd(const void* q, const void* k, const void* v, void* o,
                              int H, int S, cudaStream_t stream) {
  if ((S % BR) != 0 || DW != 32) return -1;
  const float scale = 1.f / sqrtf((float)DW);
  const float qscale = scale * 1.4426950408889634f;  // exp2-domain softmax
  dim3 grid(S / BR, H), block(128);
  flash_fwd_kernel<<<grid, block, 0, stream>>>(
      (const __half*)q, (const __half*)k, (const __half*)v, (__half*)o, H, S, qscale);
  return (int)cudaGetLastError();
}
