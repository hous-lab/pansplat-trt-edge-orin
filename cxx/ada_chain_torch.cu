#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include "so3_const.cuh"

__device__ inline void apply_e(float* v, int m, const float* X, const float* X2,
                               float lam, float th) {
  float s = sinf(lam * th) / lam;
  float c2 = (1.f - cosf(lam * th)) / (lam * lam);
  float out[9];
  for (int i = 0; i < m; ++i) {
    float xv = 0.f, x2v = 0.f;
    for (int j = 0; j < m; ++j) {
      xv += X[i * m + j] * v[j];
      x2v += X2[i * m + j] * v[j];
    }
    out[i] = v[i] + s * xv + c2 * x2v;
  }
  for (int i = 0; i < m; ++i) v[i] = out[i];
}

__device__ inline void quat_to_R(const float* q, float* R) {
  float i = q[0], j = q[1], k = q[2], r = q[3];
  float ss = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3] + 1e-8f;
  float two_s = 2.f / ss;
  R[0] = 1.f - two_s * (j*j + k*k);
  R[1] = two_s * (i*j - k*r);
  R[2] = two_s * (i*k + j*r);
  R[3] = two_s * (i*j + k*r);
  R[4] = 1.f - two_s * (i*i + k*k);
  R[5] = two_s * (j*k - i*r);
  R[6] = two_s * (i*k - j*r);
  R[7] = two_s * (j*k + i*r);
  R[8] = 1.f - two_s * (i*i + j*j);
}

__global__ void ada_kernel(
    const __half* __restrict__ raw_g, const __half* __restrict__ dc_s,
    const float* __restrict__ xy, const __half* __restrict__ nearf,
    const __half* __restrict__ farf, const __half* __restrict__ extr,
    float* __restrict__ means, float* __restrict__ cov, float* __restrict__ harm,
    float* __restrict__ opa,
    int vb, int n, float smin, float smax, float mult, float inv_w, float inv_h) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= vb * n) return;
  int bi = t / n, ni = t - bi * n;
  const __half* g = raw_g + (long)t * 117;
  const __half* dcs = dc_s + (long)t * 32;

  float pdf[32];
  float mx = -3.4e38f;
  for (int k = 0; k < 32; ++k) mx = fmaxf(mx, __half2float(g[84 + k]));
  float Z = 0.f;
  for (int k = 0; k < 32; ++k) Z += __expf(__half2float(g[84 + k]) - mx);
  float disp = 0.f;
  for (int k = 0; k < 32; ++k) {
    pdf[k] = __expf(__half2float(g[84 + k]) - mx) / Z;
    disp += pdf[k] * __half2float(dcs[k]);
  }
  float nrm_far = 1.f / __half2float(farf[bi]);
  float nrm_near = 1.f / __half2float(nearf[bi]);
  disp = fminf(fmaxf(disp, nrm_far), nrm_near);
  float depth = 1.f / disp;
  float rho = __half2float(g[116]);
  opa[t] = 1.f / (1.f + __expf(-rho));

  float xf = xy[2 * ni], yf = xy[2 * ni + 1];
  float lat = yf * CUDART_PI_F * 0.5f;
  float pw = 1.f / fmaxf(cosf(lat), 1e-2f);
  float ps_x = pw * inv_w, ps_y = inv_h;
  float o0 = 1.f / (1.f + __expf(-__half2float(g[0])));
  float o1 = 1.f / (1.f + __expf(-__half2float(g[1])));
  float xr = xf * 0.5f + 0.5f + (o0 - 0.5f) * ps_x;
  float yr = yf * 0.5f + 0.5f + (o1 - 0.5f) * ps_y;
  float phi = xr * 2.f * CUDART_PI_F - CUDART_PI_F;
  float theta = yr * CUDART_PI_F - CUDART_PI_F * 0.5f;
  float dx = cosf(theta) * sinf(phi), dy = sinf(theta), dz = cosf(theta) * cosf(phi);
  const __half* E = extr + (long)bi * 16;
  float Re[3][3];
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      Re[a][b] = __half2float(E[a * 4 + b]);
  float te[3] = {__half2float(E[3]), __half2float(E[7]), __half2float(E[11])};
  float s[3];
  for (int k = 0; k < 3; ++k) {
    float sg = 1.f / (1.f + __expf(-__half2float(g[2 + k])));
    s[k] = (smin + (smax - smin) * sg) * depth * mult;
  }
  float q[4];
  float qn = 1e-8f;
  for (int k = 0; k < 4; ++k) { q[k] = __half2float(g[5 + k]); qn += q[k]*q[k]; }
  qn = sqrtf(qn);
  for (int k = 0; k < 4; ++k) q[k] /= qn;
  float R[9];
  quat_to_R(q, R);

  float cov_c[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      cov_c[i][j] = R[i*3+0]*s[0]*s[0]*R[j*3+0] + R[i*3+1]*s[1]*s[1]*R[j*3+1] + R[i*3+2]*s[2]*s[2]*R[j*3+2];
  float cov_w[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float v = 0.f;
      for (int m = 0; m < 3; ++m)
        for (int n = 0; n < 3; ++n)
          v += Re[i][m] * cov_c[m][n] * Re[j][n];
      cov_w[i][j] = v;
    }

  for (int k = 0; k < 3; ++k)
    means[3 * t + k] = te[k] + (Re[k][0]*dx + Re[k][1]*dy + Re[k][2]*dz) * depth;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      cov[9 * t + 3 * i + j] = cov_w[i][j];
  float a = atan2f(Re[0][1], Re[2][1]);
  float b = acosf(fminf(fmaxf(Re[1][1], -1.f), 1.f));
  float ea[9], eb[9];
  float sA = sinf(LAM1Y * a) / LAM1Y, cA = (1.f - cosf(LAM1Y * a)) / (LAM1Y * LAM1Y);
  float sB = sinf(LAM1X * b) / LAM1X, cB = (1.f - cosf(LAM1X * b)) / (LAM1X * LAM1X);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      ea[i*3+j] = (i == j) + sA * X1Y[i*3+j] + cA * X1Y2[i*3+j];
      eb[i*3+j] = (i == j) + sB * X1X[i*3+j] + cB * X1X2[i*3+j];
    }
  float R1[9];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float v = 0.f;
      for (int m = 0; m < 3; ++m) v += ea[i*3+m] * eb[m*3+j];
      R1[i*3+j] = v;
    }
  float R2_00, R2_02;
  {
    float r2_02 = 0.f, r2_00 = 0.f;
    for (int m = 0; m < 3; ++m) {
      r2_02 += R1[m*3+0] * Re[m][2];
      r2_00 += R1[m*3+0] * Re[m][0];
    }
    R2_02 = r2_02; R2_00 = r2_00;
  }
  float c = atan2f(R2_02, R2_00);
  const float SM[25] = {
    1.f,
    0.025f, 0.025f, 0.025f,
    0.00625f, 0.00625f, 0.00625f, 0.00625f, 0.00625f,
    0.0015625f, 0.0015625f, 0.0015625f, 0.0015625f, 0.0015625f, 0.0015625f, 0.0015625f,
    0.000390625f, 0.000390625f, 0.000390625f, 0.000390625f, 0.000390625f,
    0.000390625f, 0.000390625f, 0.000390625f, 0.000390625f};
  for (int xyz = 0; xyz < 3; ++xyz) {
    float v[25];
    for (int d = 0; d < 25; ++d) v[d] = __half2float(g[9 + xyz * 25 + d]) * SM[d];
    for (int l = 1; l <= 4; ++l) {
      int m = 2 * l + 1, off = l * l;
      const float *XX, *XX2, *XY, *XY2;
      float lx, ly;
      if (l == 1) { XY=X1Y; XY2=X1Y2; XX=X1X; XX2=X1X2; ly=LAM1Y; lx=LAM1X; }
      else if (l == 2) { XY=X2Y; XY2=X2Y2; XX=X2X; XX2=X2X2; ly=LAM2Y; lx=LAM2X; }
      else if (l == 3) { XY=X3Y; XY2=X3Y2; XX=X3X; XX2=X3X2; ly=LAM3Y; lx=LAM3X; }
      else { XY=X4Y; XY2=X4Y2; XX=X4X; XX2=X4X2; ly=LAM4Y; lx=LAM4X; }
      apply_e(&v[off], m, XY, XY2, ly, c);
      apply_e(&v[off], m, XX, XX2, lx, b);
      apply_e(&v[off], m, XY, XY2, ly, a);
    }
    for (int d = 0; d < 25; ++d) harm[75 * t + xyz * 25 + d] = v[d];
  }
}

std::vector<torch::Tensor> ada_forward(
    torch::Tensor raw_g, torch::Tensor dc_s, torch::Tensor xy,
    torch::Tensor nref, torch::Tensor fref, torch::Tensor extr,
    double smin, double smax, double h, double w) {
  int vb = raw_g.size(0) * raw_g.size(1);
  int n = raw_g.size(2);
  auto opts = raw_g.options().dtype(torch::kFloat);
  auto means = torch::empty({1, vb * n, 3}, opts);
  auto cov = torch::empty({1, vb * n, 3, 3}, opts);
  auto harm = torch::empty({1, vb * n, 3, 25}, opts);
  auto opa = torch::empty({1, vb * n}, opts);
  int total = vb * n;
  int threads = 256;
  long tt = total;
  int blocks = (int)((tt + threads - 1) / threads);
  float mult = 0.1f * (2.f * CUDART_PI_F / (float)w + CUDART_PI_F / (float)h);
  ada_kernel<<<blocks, threads>>>(
      (const __half*)raw_g.data_ptr<at::Half>(),
      (const __half*)dc_s.data_ptr<at::Half>(),
      xy.data_ptr<float>(),
      (const __half*)nref.data_ptr<at::Half>(),
      (const __half*)fref.data_ptr<at::Half>(),
      (const __half*)extr.data_ptr<at::Half>(),
      means.data_ptr<float>(), cov.data_ptr<float>(),
      harm.data_ptr<float>(), opa.data_ptr<float>(),
      vb, n, (float)smin, (float)smax, mult, 1.f / (float)w, 1.f / (float)h);
  return {means, cov, harm, opa};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ada_forward", &ada_forward, "fused GHAdapterSeg (fp16 in, fp32 out)");
}
