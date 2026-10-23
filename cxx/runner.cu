// GPU-resident runner: 21 TRT FP16 engines + 5 custom CUDA kernel classes
// (41 kernel launches, 31 timed segments), zero file-chain.
// Build: make -C cxx runner  (auto-detects sm arch and TRT lib dir)
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <cassert>
#include <algorithm>
#include <memory>
#include "so3_const.cuh"
#include "flash_attn.cu"   // flash attention (exp_flash/flash_attn.cu)

#define CK(x) do { auto e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)

struct Logger : public nvinfer1::ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) fprintf(stderr, "[TRT] %s\n", msg);
  }
};
static Logger gLogger;

struct Buf {
  void* d = nullptr;
  size_t nbytes = 0;
};

static std::unordered_map<std::string, Buf> g_bufs;

static void alloc_buf(const std::string& name, size_t nbytes) {
  if (g_bufs.count(name)) return;
  void* p = nullptr;
  CK(cudaMalloc(&p, nbytes));
  g_bufs[name] = {p, nbytes};
}

static std::vector<char> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  assert(f.good());
  f.seekg(0, std::ios::end);
  size_t n = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<char> v(n);
  f.read(v.data(), n);
  return v;
}

static void* load_bin(const std::string& path, size_t expected_bytes) {
  std::vector<char> v = read_file(path);
  if (v.size() != expected_bytes) {
    fprintf(stderr, "bin size mismatch %s: %zu vs %zu\n", path.c_str(), v.size(), expected_bytes);
    exit(1);
  }
  void* p = nullptr;
  CK(cudaMalloc(&p, v.size()));
  CK(cudaMemcpy(p, v.data(), v.size(), cudaMemcpyHostToDevice));
  return p;
}

static void dump_dev(const void* dptr, size_t nbytes, const char* path) {
  std::vector<char> h(nbytes);
  CK(cudaMemcpy(h.data(), dptr, nbytes, cudaMemcpyDeviceToHost));
  FILE* fp = fopen(path, "wb");
  fwrite(h.data(), 1, nbytes, fp);
  fclose(fp);
}

static float ms_of(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0.f;
  cudaEventElapsedTime(&ms, a, b);
  return ms;
}

// ---- ada kernel device code (single source: cxx/ada_chain_torch.cu) ----

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
    const __half* __restrict__ shb, int has_shb,
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
    if (has_shb) v[0] += 2.f * (__half2float(shb[3 * t + xyz]) - 0.5f);
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


static void ada_launch_c(const void* raw_g, const void* dc_s, const void* xy,
                         const void* nref, const void* fref, const void* extr,
                         const void* shb, int has_shb,
                         float* means, float* cov, float* harm, float* opa,
                         int vb, int n, float smin, float smax, float h, float w,
                         cudaStream_t stream) {
  float mult = 0.1f * (2.f * CUDART_PI_F / w + CUDART_PI_F / h);
  int threads = 256;
  long tt = (long)vb * n;
  int blocks = (int)((tt + threads - 1) / threads);
  ada_kernel<<<blocks, threads, 0, stream>>>(
      (const __half*)raw_g, (const __half*)dc_s, (const float*)xy,
      (const __half*)nref, (const __half*)fref, (const __half*)extr,
      (const __half*)shb, has_shb,
      means, cov, harm, opa, vb, n, smin, smax, mult, 1.f / w, 1.f / h);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) fprintf(stderr, "ADA LAUNCH FAIL: %s\n", cudaGetErrorString(err));
}

// ---- CV kernel (single source: cxx/cv_kernel_torch.cu) ----
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

static void cv_launch(const void* fref, const void* fsrc, const void* dvol,
                      const void* pose, void* corr,
                      int vb, int C, int D, int H, int W, cudaStream_t stream) {
  int threads = 256;
  long tt = (long)vb * D * H * W;
  int blocks = (int)((tt + threads - 1) / threads);
  cost_volume_kernel<<<blocks, threads, 0, stream>>>(
      (const __half*)fref, (const __half*)fsrc, (const float*)dvol,
      (const float*)pose, (__half*)corr, vb, C, D, H, W,
      1.f / sqrtf((float)C));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) fprintf(stderr, "CV LAUNCH FAIL: %s\n", cudaGetErrorString(err));
}

__global__ void cast_f16_f32_kernel(const __half* src, float* dst, long n) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  dst[i] = __half2float(src[i]);
}

static void cast_f16_f32(const void* src, void* dst, long n, cudaStream_t s) {
  int blocks = (int)((n + 255) / 256);
  cast_f16_f32_kernel<<<blocks, 256, 0, s>>>((const __half*)src, (float*)dst, n);
}

__global__ void cast_f32_f16_kernel(const float* src, __half* dst, long n) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  dst[i] = __float2half(src[i]);
}

static void cast_f32_f16(const void* src, void* dst, long n, cudaStream_t s) {
  long threads = 256;
  int blocks = (int)((n + 255) / 256);
  cast_f32_f16_kernel<<<blocks, threads, 0, s>>>((const float*)src, (__half*)dst, n);
}
// ---- engines ----
struct Eng {
  std::string name;
  void* data = nullptr;
  std::shared_ptr<nvinfer1::ICudaEngine> engine;
  std::shared_ptr<nvinfer1::IExecutionContext> ctx;
};
static std::vector<Eng> g_engs;

static void load_engine(const std::string& name, const std::string& path) {
  auto blob = read_file(path);
  static nvinfer1::IRuntime* rt = nvinfer1::createInferRuntime(gLogger);
  Eng e;
  e.name = name;
  e.engine.reset(rt->deserializeCudaEngine(blob.data(), blob.size()));
  assert(e.engine != nullptr);
  e.ctx.reset(e.engine->createExecutionContext());
  g_engs.push_back(e);
  printf("engine %s loaded (%zu bytes)\n", name.c_str(), blob.size());
}
static void bind_io(Eng& e, const std::unordered_map<std::string, std::string>& wiring) {
  for (int i = 0; i < e.engine->getNbIOTensors(); ++i) {
    const char* nm = e.engine->getIOTensorName(i);
    std::string n(nm);
    if (e.engine->getTensorIOMode(nm) != nvinfer1::TensorIOMode::kINPUT) continue;
    auto it = wiring.find(n);
    if (it == wiring.end()) {
      fprintf(stderr, "unwired input %s on engine %s\n", n.c_str(), e.name.c_str());
      exit(1);
    }
    auto b = g_bufs.find(it->second);
    if (b == g_bufs.end() || b->second.d == nullptr) {
      fprintf(stderr, "no buffer %s for input %s on engine %s\n",
              it->second.c_str(), n.c_str(), e.name.c_str());
      exit(1);
    }
    if (!e.ctx->setTensorAddress(nm, b->second.d))
      fprintf(stderr, "BINDFAIL in %s <- %s on %s\n", nm, it->second.c_str(), e.name.c_str());
  }
}

static void bind_out_as(Eng& e, const std::unordered_map<std::string, std::string>& m) {
  for (int i = 0; i < e.engine->getNbIOTensors(); ++i) {
    const char* nm = e.engine->getIOTensorName(i);
    if (e.engine->getTensorIOMode(nm) != nvinfer1::TensorIOMode::kOUTPUT) continue;
    auto it = m.find(std::string(nm));
    if (it == m.end()) continue;
    if (!e.ctx->setTensorAddress(nm, g_bufs[it->second].d))
      fprintf(stderr, "BINDFAIL out %s -> %s on %s\n", nm, it->second.c_str(), e.name.c_str());
  }
}

static void bind_out(Eng& e) {
  for (int i = 0; i < e.engine->getNbIOTensors(); ++i) {
    const char* nm = e.engine->getIOTensorName(i);
    if (e.engine->getTensorIOMode(nm) != nvinfer1::TensorIOMode::kOUTPUT) continue;
    std::string n(nm);
    auto it = g_bufs.find(n);
    if (it == g_bufs.end()) {
      fprintf(stderr, "no buffer for output %s on engine %s\n", n.c_str(), e.name.c_str());
      exit(1);
    }
    if (!e.ctx->setTensorAddress(nm, it->second.d))
      fprintf(stderr, "BINDFAIL out* %s on %s\n", nm, e.name.c_str());
  }
}
int main(int argc, char** argv) {
  const char* bin_dir = argc > 1 ? argv[1] : "export/runner_bins";
  const char* eng_dir = argc > 2 ? argv[2] : "export/engines";
  const char* out_path = argc > 3 ? argv[3] : "export/scratch/gaussians_runner.bin";
  cudaFree(0);
  // ---- buffer table: name -> total elements (f32 unless suffix h = fp16) ----
  struct BDef { const char* name; long nel; };
  const BDef defs[] = {
    {"image", 1*2*3*512*1024L}, {"mono_image", 1*2*3*512*1024L},
    {"cube_image", 1*2*3*256*1536L}, {"extrinsics", 1*2*4*4L},
    {"near", 2L}, {"far", 2L},
    {"feat_0", 1*2*128*64*128L}, {"feat_1", 1*2*96*128*256L},
    {"feat_2", 1*2*64*256*512L}, {"feat_3", 1*2*32*512*1024L},
    {"raw_correlation", 1*2*32*256*512L}, {"disp_candi_curr", 1*2*32*256*512L},
    {"image_pass", 1*2*3*512*1024L}, {"extrinsics_pass", 1*2*4*4L},
    {"near_pass", 2L}, {"far_pass", 2L},
    {"last_raw_0", 2*234*70*134L}, {"last_raw_1", 2*234*134*262L},
    {"raw_cnn_2", 2*234*262*518L}, {"images_c_2", 2*3*262*518L},
    {"raw_cnn_3", 2*234*518*1030L}, {"images_c_3", 2*3*518*1030L},
    {"raw_g_2", 1*2*83443*117L}, {"dc_s_2", 1*2*83443*32L}, {"sh_base_2", 1L},
    {"raw_g_3", 1*2*333772*117L}, {"dc_s_3", 1*2*333772*32L}, {"sh_base_3", 1L},
    {"raw_g_2h", 1*2*83443*117L}, {"dc_s_2h", 1*2*83443*32L},
    {"raw_g_3h", 1*2*333772*117L}, {"dc_s_3h", 1*2*333772*32L},
    {"xy_s2", 83443*2L}, {"xy_s3", 333772*2L},
    {"means_0", 10430L*3}, {"covariances_0", 10430L*9},
    {"harmonics_0", 10430L*75}, {"opacities_0", 10430L},
    {"means_1", 41720L*3}, {"covariances_1", 41720L*9},
    {"harmonics_1", 41720L*75}, {"opacities_1", 41720L},
    {"g2_means", 166886L*3}, {"g2_cov", 166886L*9},
    {"g2_harm", 166886L*75}, {"g2_opa", 166886L},
    {"g3_means", 667544L*3}, {"g3_cov", 667544L*9},
    {"g3_harm", 667544L*75}, {"g3_opa", 667544L},
    {"final_means", 886580L*3}, {"final_cov", 886580L*9},
    {"final_harm", 886580L*75}, {"final_opa", 886580L},
    {"feat01_0", 2*128*64*128L}, {"feat10_0", 2*128*64*128L},
    {"dvol_0", 2*128*64*128L}, {"dc0", 2*128*64*128L}, {"corr0", 2*128*64*128L},
    {"disp0", 2*1*64*128L}, {"feat01_0h", 2*128*64*128L}, {"feat10_0h", 2*128*64*128L},
    {"feat01_1", 2*96*128*256L}, {"feat10_1", 2*96*128*256L},
    {"dvol_1", 2*64*128*256L}, {"dc1", 2*64*128*256L}, {"corr1", 2*64*128*256L},
    {"disp1", 2*1*128*256L}, {"feat01_1h", 2*96*128*256L}, {"feat10_1h", 2*96*128*256L},
    {"feat01_2", 2*64*256*512L}, {"feat10_2", 2*64*256*512L},
    {"dvol_2", 2*32*256*512L}, {"dc2", 2*32*256*512L}, {"corr2", 2*32*256*512L},
    {"disp2", 2*1*256*512L}, {"feat01_2h", 2*64*256*512L}, {"feat10_2h", 2*64*256*512L},
    {"mono_feat", 1*2*32*256*512L},
    {"disp_candi", 2*32*256*512L},
    {"pose", 2*16L},
    {"corr0h", 2*128*64*128L}, {"corr1h", 2*64*128*256L}, {"corr2h", 2*32*256*512L},
    {"images_c_0", 2*3*70*134L}, {"images_c_1", 2*3*134*262L},
    {"raw_g_0", 1*2*5215*117L}, {"dc_s_0", 1*2*5215*32L}, {"sh_base_0", 1*2*5215*3L},
    {"raw_g_0h", 1*2*5215*117L}, {"dc_s_0h", 1*2*5215*32L}, {"shb_0h", 1*2*5215*3L},
    {"raw_g_1", 1*2*20860*117L}, {"dc_s_1", 1*2*20860*32L}, {"sh_base_1", 1L},
    {"raw_g_1h", 1*2*20860*117L}, {"dc_s_1h", 1*2*20860*32L},
    {"xy_s0", 5215*2L}, {"xy_s1", 20860*2L},
    // flash-attention split boundaries (fp16 q/k/v/o + fp32 UNet skips)
    {"p1_q1h", 2*4096*32L}, {"p1_k1h", 2*4096*32L}, {"p1_v1h", 2*4096*32L},
    {"p1_o1h", 2*4096*32L},
    {"p1_q2h", 2*4096*32L}, {"p1_k2h", 2*4096*32L}, {"p1_v2h", 2*4096*32L},
    {"p1_o2h", 2*4096*32L},
    {"p1_q3h", 2*4096*32L}, {"p1_k3h", 2*4096*32L}, {"p1_v3h", 2*4096*32L},
    {"p1_o3h", 2*4096*32L},
    {"p1_sk1", 2*193*128*256L}, {"p1_sk2", 2*64*128*256L},
    {"p1_sk3", 2*64*128*256L}, {"p1_sk4", 2*64*64*128L},
    {"p1_sk5", 2*64*64*128L}, {"p1_sk6", 2*64*32*64L},
    {"p1_sk7", 2*64*2048L}, {"p1_sk8", 2*64*2048L}, {"p1_sk9", 2*64*2048L},
    {"p2_q1h", 1*16384*32L}, {"p2_k1h", 1*16384*32L}, {"p2_v1h", 1*16384*32L},
    {"p2_o1h", 1*16384*32L},
    {"p2_q2h", 1*16384*32L}, {"p2_k2h", 1*16384*32L}, {"p2_v2h", 1*16384*32L},
    {"p2_o2h", 1*16384*32L},
    {"p2_q3h", 1*16384*32L}, {"p2_k3h", 1*16384*32L}, {"p2_v3h", 1*16384*32L},
    {"p2_o3h", 1*16384*32L},
    {"p2_sk1", 2*129*256*512L}, {"p2_sk2", 2*32*256*512L},
    {"p2_sk3", 2*32*256*512L}, {"p2_sk4", 2*32*128*256L},
    {"p2_sk5", 2*32*128*256L}, {"p2_sk6", 2*32*64*128L},
    {"p2_sk7", 2*32*8192L}, {"p2_sk8", 2*32*8192L}, {"p2_sk9", 2*32*8192L},
  };
  for (auto& d : defs) alloc_buf(d.name, d.nel * 4);
  // ---- load inputs ----
  g_bufs["image"].d = load_bin(std::string(bin_dir) + "/image.bin", 1*2*3*512*1024L*4);
  g_bufs["mono_image"].d = load_bin(std::string(bin_dir) + "/mono_image.bin", 1*2*3*512*1024L*4);
  g_bufs["cube_image"].d = load_bin(std::string(bin_dir) + "/cube_image.bin", 1*2*3*256*1536L*4);
  g_bufs["extrinsics"].d = load_bin(std::string(bin_dir) + "/extrinsics.bin", 1*2*4*4L*4);
  g_bufs["near"].d = load_bin(std::string(bin_dir) + "/near.bin", 8);
  g_bufs["far"].d = load_bin(std::string(bin_dir) + "/far.bin", 2*4);
  g_bufs["xy_s2"].d = load_bin(std::string(bin_dir) + "/xy_s2.bin", 83443*2L*4);
  g_bufs["xy_s3"].d = load_bin(std::string(bin_dir) + "/xy_s3.bin", 333772*2L*4);
  g_bufs["xy_s0"].d = load_bin(std::string(bin_dir) + "/xy_s0.bin", 5215*2L*4);
  g_bufs["xy_s1"].d = load_bin(std::string(bin_dir) + "/xy_s1.bin", 20860*2L*4);
  printf("inputs loaded\n");
  // ---- engines + wiring ----
  load_engine("mvs_pre", std::string(eng_dir) + "/mvs_pre.engine");
  load_engine("cv_pre0", std::string(eng_dir) + "/cv_pre0.engine");
  load_engine("cv_post0", std::string(eng_dir) + "/cv_post0.engine");
  load_engine("cv_pre1", std::string(eng_dir) + "/cv_pre1.engine");
  load_engine("cv_post1a", std::string(eng_dir) + "/p1_a.engine");
  load_engine("cv_post1m1", std::string(eng_dir) + "/p1_m1.engine");
  load_engine("cv_post1m2", std::string(eng_dir) + "/p1_m2.engine");
  load_engine("cv_post1b", std::string(eng_dir) + "/p1_b.engine");
  load_engine("cv_pre2", std::string(eng_dir) + "/cv_pre2.engine");
  load_engine("cv_post2a", std::string(eng_dir) + "/p2_a.engine");
  load_engine("cv_post2m1", std::string(eng_dir) + "/p2_m1.engine");
  load_engine("cv_post2m2", std::string(eng_dir) + "/p2_m2.engine");
  load_engine("cv_post2b", std::string(eng_dir) + "/p2_b.engine");
  load_engine("gh_s0_cnn", std::string(eng_dir) + "/gh_s0_cnn.engine");
  load_engine("gh_s0_mlp", std::string(eng_dir) + "/gh_s0_mlp.engine");
  load_engine("gh_s1_cnn", std::string(eng_dir) + "/gh_s1_cnn.engine");
  load_engine("gh_s1_mlp", std::string(eng_dir) + "/gh_s1_mlp.engine");
  load_engine("gh_s2_cnn", std::string(eng_dir) + "/gh_s2_cnn.engine");
  load_engine("gh_s2_mlp", std::string(eng_dir) + "/gh_s2_mlp.engine");
  load_engine("gh_s3_cnn", std::string(eng_dir) + "/gh_s3_cnn.engine");
  load_engine("gh_s3_mlp", std::string(eng_dir) + "/gh_s3_mlp.engine");

  auto eng = [&](const char* k) -> Eng& {
    for (auto& e : g_engs) if (e.name == k) return e;
    fprintf(stderr, "no engine %s\n", k);
    exit(1);
  };

  bind_io(eng("mvs_pre"), {{"image","image"},{"extrinsics","extrinsics"},{"near","near"},{"far","far"},
                       {"mono_image","mono_image"},{"cube_image","cube_image"}});
  bind_out(eng("mvs_pre"));
  bind_io(eng("cv_pre0"), {{"features","feat_0"},{"extrinsics","extrinsics"},
                           {"near","near"},{"far","far"}});
  bind_out_as(eng("cv_pre0"), {{"feat01","feat01_0"},{"feat10","feat10_0"},
      {"depth_vol","dvol_0"},{"pose","pose"},{"disp_candi","dc0"}});
  bind_io(eng("cv_post0"), {{"corr","corr0"},{"feat01","feat01_0"},{"mono_feat","mono_feat"},
                            {"disp_candi","dc0"}});
  bind_out_as(eng("cv_post0"), {{"disp_out","disp0"}});
  bind_io(eng("cv_pre1"), {{"features","feat_1"},{"extrinsics","extrinsics"},
                           {"near","near"},{"far","far"},{"coarse_disp","disp0"}});
  bind_out_as(eng("cv_pre1"), {{"feat01","feat01_1"},{"feat10","feat10_1"},
      {"depth_vol","dvol_1"},{"pose","pose"},{"disp_candi","dc1"}});
  // ---- cv_post1 split (flash attention between segments) ----
  bind_io(eng("cv_post1a"), {{"corr","corr1"},{"feat01","feat01_1"},
                             {"mono_feat","mono_feat"},{"coarse_disp","disp0"}});
  bind_out_as(eng("cv_post1a"), {
      {"/Concat_2_output_0","p1_sk1"},
      {"/refine/refine.3/input_blocks.0/input_blocks.0.0/Conv_output_0","p1_sk2"},
      {"/refine/refine.3/input_blocks.1/input_blocks.1.0/Add_output_0","p1_sk3"},
      {"/refine/refine.3/input_blocks.2/input_blocks.2.0/op/Conv_output_0","p1_sk4"},
      {"/refine/refine.3/input_blocks.3/input_blocks.3.0/Add_output_0","p1_sk5"},
      {"/refine/refine.3/input_blocks.4/input_blocks.4.0/op/Conv_output_0","p1_sk6"},
      {"/refine/refine.3/input_blocks.5/input_blocks.5.1/Reshape_output_0","p1_sk7"},
      {"p1_q1","p1_q1h"},{"p1_k1","p1_k1h"},{"p1_v1","p1_v1h"}});
  bind_io(eng("cv_post1m1"), {
      {"/refine/refine.3/input_blocks.5/input_blocks.5.1/Reshape_output_0","p1_sk7"},
      {"p1_o1","p1_o1h"}});
  bind_out_as(eng("cv_post1m1"), {
      {"/refine/refine.3/output_blocks.0/output_blocks.0.1/Reshape_output_0","p1_sk8"},
      {"p1_q2","p1_q2h"},{"p1_k2","p1_k2h"},{"p1_v2","p1_v2h"}});
  bind_io(eng("cv_post1m2"), {
      {"/refine/refine.3/input_blocks.4/input_blocks.4.0/op/Conv_output_0","p1_sk6"},
      {"/refine/refine.3/output_blocks.0/output_blocks.0.1/Reshape_output_0","p1_sk8"},
      {"p1_o2","p1_o2h"}});
  bind_out_as(eng("cv_post1m2"), {
      {"/refine/refine.3/output_blocks.1/output_blocks.1.1/Reshape_output_0","p1_sk9"},
      {"p1_q3","p1_q3h"},{"p1_k3","p1_k3h"},{"p1_v3","p1_v3h"}});
  bind_io(eng("cv_post1b"), {
      {"/Concat_2_output_0","p1_sk1"},
      {"/refine/refine.3/input_blocks.0/input_blocks.0.0/Conv_output_0","p1_sk2"},
      {"/refine/refine.3/input_blocks.1/input_blocks.1.0/Add_output_0","p1_sk3"},
      {"/refine/refine.3/input_blocks.2/input_blocks.2.0/op/Conv_output_0","p1_sk4"},
      {"/refine/refine.3/input_blocks.3/input_blocks.3.0/Add_output_0","p1_sk5"},
      {"/refine/refine.3/output_blocks.1/output_blocks.1.1/Reshape_output_0","p1_sk9"},
      {"disp_candi","dc1"},{"p1_o3","p1_o3h"}});
  bind_out_as(eng("cv_post1b"), {{"disp_out","disp1"}});
  bind_io(eng("cv_pre2"), {{"features","feat_2"},{"extrinsics","extrinsics"},
                           {"near","near"},{"far","far"},{"coarse_disp","disp1"}});
  bind_out_as(eng("cv_pre2"), {{"feat01","feat01_2"},{"feat10","feat10_2"},
      {"depth_vol","dvol_2"},{"pose","pose"},{"disp_candi","dc2"}});
  // ---- cv_post2 split (flash attention between segments) ----
  bind_io(eng("cv_post2a"), {{"corr","corr2"},{"feat01","feat01_2"},
                             {"mono_feat","mono_feat"},{"coarse_disp","disp1"}});
  bind_out_as(eng("cv_post2a"), {
      {"/Concat_2_output_0","p2_sk1"},
      {"/refine/refine.3/input_blocks.0/input_blocks.0.0/Conv_output_0","p2_sk2"},
      {"/refine/refine.3/input_blocks.1/input_blocks.1.0/Add_output_0","p2_sk3"},
      {"/refine/refine.3/input_blocks.2/input_blocks.2.0/op/Conv_output_0","p2_sk4"},
      {"/refine/refine.3/input_blocks.3/input_blocks.3.0/Add_output_0","p2_sk5"},
      {"/refine/refine.3/input_blocks.4/input_blocks.4.0/op/Conv_output_0","p2_sk6"},
      {"/refine/refine.3/input_blocks.5/input_blocks.5.1/Reshape_output_0","p2_sk7"},
      {"p2_q1","p2_q1h"},{"p2_k1","p2_k1h"},{"p2_v1","p2_v1h"}});
  bind_io(eng("cv_post2m1"), {
      {"/refine/refine.3/input_blocks.5/input_blocks.5.1/Reshape_output_0","p2_sk7"},
      {"p2_o1","p2_o1h"}});
  bind_out_as(eng("cv_post2m1"), {
      {"/refine/refine.3/output_blocks.0/output_blocks.0.1/Reshape_output_0","p2_sk8"},
      {"p2_q2","p2_q2h"},{"p2_k2","p2_k2h"},{"p2_v2","p2_v2h"}});
  bind_io(eng("cv_post2m2"), {
      {"/refine/refine.3/input_blocks.4/input_blocks.4.0/op/Conv_output_0","p2_sk6"},
      {"/refine/refine.3/output_blocks.0/output_blocks.0.1/Reshape_output_0","p2_sk8"},
      {"p2_o2","p2_o2h"}});
  bind_out_as(eng("cv_post2m2"), {
      {"/refine/refine.3/output_blocks.1/output_blocks.1.1/Reshape_output_0","p2_sk9"},
      {"p2_q3","p2_q3h"},{"p2_k3","p2_k3h"},{"p2_v3","p2_v3h"}});
  bind_io(eng("cv_post2b"), {
      {"/Concat_2_output_0","p2_sk1"},
      {"/refine/refine.3/input_blocks.0/input_blocks.0.0/Conv_output_0","p2_sk2"},
      {"/refine/refine.3/input_blocks.1/input_blocks.1.0/Add_output_0","p2_sk3"},
      {"/refine/refine.3/input_blocks.2/input_blocks.2.0/op/Conv_output_0","p2_sk4"},
      {"/refine/refine.3/input_blocks.3/input_blocks.3.0/Add_output_0","p2_sk5"},
      {"/refine/refine.3/output_blocks.1/output_blocks.1.1/Reshape_output_0","p2_sk9"},
      {"disp_candi","dc2"},{"p2_o3","p2_o3h"}});
  bind_out_as(eng("cv_post2b"), {{"disp_out","disp2"}});
  bind_io(eng("gh_s0_cnn"), {});
  bind_out_as(eng("gh_s0_cnn"), {{"raw_cnn_0","last_raw_0"},{"images_c_0","images_c_0"}});
  bind_io(eng("gh_s0_mlp"), {{"raw_cnn","last_raw_0"},{"images_c","images_c_0"},
                             {"disp_candi_curr","dc2"}});
  bind_out(eng("gh_s0_mlp"));
  bind_io(eng("gh_s1_cnn"), {{"last_raw","last_raw_0"}});
  bind_out_as(eng("gh_s1_cnn"), {{"raw_cnn_1","last_raw_1"},{"images_c_1","images_c_1"}});
  bind_io(eng("gh_s1_mlp"), {{"raw_cnn","last_raw_1"},{"disp_candi_curr","dc2"}});
  bind_out(eng("gh_s1_mlp"));
  bind_io(eng("gh_s2_cnn"), {{"last_raw","last_raw_1"}});
  bind_out(eng("gh_s2_cnn"));
  bind_io(eng("gh_s2_mlp"), {{"raw_cnn","raw_cnn_2"},{"disp_candi_curr","dc2"}});
  bind_out(eng("gh_s2_mlp"));
  bind_io(eng("gh_s3_cnn"), {{"last_raw","raw_cnn_2"}});
  bind_out(eng("gh_s3_cnn"));
  bind_io(eng("gh_s3_mlp"), {{"raw_cnn","raw_cnn_3"},{"disp_candi_curr","dc2"}});
  bind_out(eng("gh_s3_mlp"));
  printf("engines wired\n");
  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  // ---- extra fp16 buffers for kernels ----
  void* near_h = nullptr; CK(cudaMalloc(&near_h, 8));
  void* far_h = nullptr; CK(cudaMalloc(&far_h, 8));
  void* extr_h = nullptr; CK(cudaMalloc(&extr_h, 128));
  // ---- timing events ----
  const int NS = 31;
  cudaEvent_t ev0[NS], ev1[NS];
  for (int i = 0; i < NS; ++i) { CK(cudaEventCreate(&ev0[i])); CK(cudaEventCreate(&ev1[i])); }

  auto chain_once = [&](bool timing) {
    auto seg = [&](int i) { if (timing) CK(cudaEventRecord(ev0[i], s)); };
    auto segend = [&](int i) { if (timing) CK(cudaEventRecord(ev1[i], s)); };
    cast_f32_f16(g_bufs["near"].d, near_h, 2, s);
    cast_f32_f16(g_bufs["far"].d, far_h, 2, s);
    cast_f32_f16(g_bufs["extrinsics"].d, extr_h, 32, s);
    seg(0);
    if (!eng("mvs_pre").ctx->enqueueV3(s)) fprintf(stderr, "ENQUEUE FAIL\n");
    segend(0);
    seg(1);
    if (!eng("cv_pre0").ctx->enqueueV3(s)) fprintf(stderr, "ENQUEUE FAIL\n");
    segend(1);
    seg(2);
    cast_f32_f16(g_bufs["feat01_0"].d, g_bufs["feat01_0h"].d, 2*128*64*128L, s);
    cast_f32_f16(g_bufs["feat10_0"].d, g_bufs["feat10_0h"].d, 2*128*64*128L, s);
    cv_launch(g_bufs["feat01_0h"].d, g_bufs["feat10_0h"].d, g_bufs["dvol_0"].d,
              g_bufs["pose"].d, g_bufs["corr0h"].d, 2, 128, 128, 64, 128, s);
    cast_f16_f32(g_bufs["corr0h"].d, g_bufs["corr0"].d, 2*128*64*128L, s);
    segend(2);
    seg(3);
    if (!eng("cv_post0").ctx->enqueueV3(s)) fprintf(stderr, "ENQUEUE FAIL\n");
    segend(3);
    seg(4);
    if (!eng("cv_pre1").ctx->enqueueV3(s)) fprintf(stderr, "ENQUEUE FAIL\n");
    segend(4);
    seg(5);
    cast_f32_f16(g_bufs["feat01_1"].d, g_bufs["feat01_1h"].d, 2*96*128*256L, s);
    cast_f32_f16(g_bufs["feat10_1"].d, g_bufs["feat10_1h"].d, 2*96*128*256L, s);
    cv_launch(g_bufs["feat01_1h"].d, g_bufs["feat10_1h"].d, g_bufs["dvol_1"].d,
              g_bufs["pose"].d, g_bufs["corr1h"].d, 2, 96, 64, 128, 256, s);
    cast_f16_f32(g_bufs["corr1h"].d, g_bufs["corr1"].d, 2*64*128*256L, s);
    segend(5);
    seg(6);
    if (!eng("cv_post1a").ctx->enqueueV3(s)) fprintf(stderr, "ENQUEUE FAIL\n");
    segend(6);
    seg(7);
    if (flash_attn_fwd(g_bufs["p1_q1h"].d, g_bufs["p1_k1h"].d, g_bufs["p1_v1h"].d,
                       g_bufs["p1_o1h"].d, 2, 4096, s))
      fprintf(stderr, "FLASH FAIL\n");
    segend(7);
    seg(8);
    if (!eng("cv_post1m1").ctx->enqueueV3(s)) fprintf(stderr, "ENQUEUE FAIL\n");
    segend(8);
    seg(9);
    if (flash_attn_fwd(g_bufs["p1_q2h"].d, g_bufs["p1_k2h"].d, g_bufs["p1_v2h"].d,
                       g_bufs["p1_o2h"].d, 2, 4096, s))
      fprintf(stderr, "FLASH FAIL\n");
    segend(9);
    seg(10);
    if (!eng("cv_post1m2").ctx->enqueueV3(s)) fprintf(stderr, "ENQUEUE FAIL\n");
    segend(10);
    seg(11);
    if (flash_attn_fwd(g_bufs["p1_q3h"].d, g_bufs["p1_k3h"].d, g_bufs["p1_v3h"].d,
                       g_bufs["p1_o3h"].d, 2, 4096, s))
      fprintf(stderr, "FLASH FAIL\n");
    segend(11);
    seg(12);
    if (!eng("cv_post1b").ctx->enqueueV3(s)) fprintf(stderr, "ENQUEUE FAIL\n");
    segend(12);
    seg(13);
    if (!eng("cv_pre2").ctx->enqueueV3(s)) fprintf(stderr, "ENQUEUE FAIL\n");
    segend(13);
    seg(14);
    cast_f32_f16(g_bufs["feat01_2"].d, g_bufs["feat01_2h"].d, 2*64*256*512L, s);
    cast_f32_f16(g_bufs["feat10_2"].d, g_bufs["feat10_2h"].d, 2*64*256*512L, s);
    cv_launch(g_bufs["feat01_2h"].d, g_bufs["feat10_2h"].d, g_bufs["dvol_2"].d,
              g_bufs["pose"].d, g_bufs["corr2h"].d, 2, 64, 32, 256, 512, s);
    cast_f16_f32(g_bufs["corr2h"].d, g_bufs["corr2"].d, 2*32*256*512L, s);
    segend(14);
    seg(15);
    if (!eng("cv_post2a").ctx->enqueueV3(s)) fprintf(stderr, "ENQUEUE FAIL\n");
    segend(15);
    seg(16);
    if (flash_attn_fwd(g_bufs["p2_q1h"].d, g_bufs["p2_k1h"].d, g_bufs["p2_v1h"].d,
                       g_bufs["p2_o1h"].d, 1, 16384, s))
      fprintf(stderr, "FLASH FAIL\n");
    segend(16);
    seg(17);
    if (!eng("cv_post2m1").ctx->enqueueV3(s)) fprintf(stderr, "ENQUEUE FAIL\n");
    segend(17);
    seg(18);
    if (flash_attn_fwd(g_bufs["p2_q2h"].d, g_bufs["p2_k2h"].d, g_bufs["p2_v2h"].d,
                       g_bufs["p2_o2h"].d, 1, 16384, s))
      fprintf(stderr, "FLASH FAIL\n");
    segend(18);
    seg(19);
    if (!eng("cv_post2m2").ctx->enqueueV3(s)) fprintf(stderr, "ENQUEUE FAIL\n");
    segend(19);
    seg(20);
    if (flash_attn_fwd(g_bufs["p2_q3h"].d, g_bufs["p2_k3h"].d, g_bufs["p2_v3h"].d,
                       g_bufs["p2_o3h"].d, 1, 16384, s))
      fprintf(stderr, "FLASH FAIL\n");
    segend(20);
    seg(21);
    if (!eng("cv_post2b").ctx->enqueueV3(s)) fprintf(stderr, "ENQUEUE FAIL\n");
    segend(21);
    seg(22);
    if (!eng("gh_s0_cnn").ctx->enqueueV3(s)) fprintf(stderr, "ENQUEUE FAIL\n");
    segend(22);
    seg(23);
    if (!eng("gh_s0_mlp").ctx->enqueueV3(s)) fprintf(stderr, "ENQUEUE FAIL\n");
    cast_f32_f16(g_bufs["raw_g_0"].d, g_bufs["raw_g_0h"].d, 1*2*5215*117L, s);
    cast_f32_f16(g_bufs["dc_s_0"].d, g_bufs["dc_s_0h"].d, 1*2*5215*32L, s);
    cast_f32_f16(g_bufs["sh_base_0"].d, g_bufs["shb_0h"].d, 1*2*5215*3L, s);
    ada_launch_c(g_bufs["raw_g_0h"].d, g_bufs["dc_s_0h"].d, g_bufs["xy_s0"].d,
                 near_h, far_h, extr_h, g_bufs["shb_0h"].d, 1,
                 (float*)g_bufs["means_0"].d, (float*)g_bufs["covariances_0"].d,
                 (float*)g_bufs["harmonics_0"].d, (float*)g_bufs["opacities_0"].d,
                 2, 5215, 0.5f, 15.f, 64.f, 128.f, s);
    segend(23);
    seg(24);
    if (!eng("gh_s1_cnn").ctx->enqueueV3(s)) fprintf(stderr, "ENQUEUE FAIL\n");
    segend(24);
    seg(25);
    if (!eng("gh_s1_mlp").ctx->enqueueV3(s)) fprintf(stderr, "ENQUEUE FAIL\n");
    cast_f32_f16(g_bufs["raw_g_1"].d, g_bufs["raw_g_1h"].d, 1*2*20860*117L, s);
    cast_f32_f16(g_bufs["dc_s_1"].d, g_bufs["dc_s_1h"].d, 1*2*20860*32L, s);
    ada_launch_c(g_bufs["raw_g_1h"].d, g_bufs["dc_s_1h"].d, g_bufs["xy_s1"].d,
                 near_h, far_h, extr_h, extr_h, 0,
                 (float*)g_bufs["means_1"].d, (float*)g_bufs["covariances_1"].d,
                 (float*)g_bufs["harmonics_1"].d, (float*)g_bufs["opacities_1"].d,
                 2, 20860, 0.5f, 15.f, 128.f, 256.f, s);
    segend(25);
    seg(26);
    if (!eng("gh_s2_cnn").ctx->enqueueV3(s)) fprintf(stderr, "ENQUEUE FAIL\n");
    segend(26);
    seg(27);
    if (!eng("gh_s2_mlp").ctx->enqueueV3(s)) fprintf(stderr, "ENQUEUE FAIL\n");
    cast_f32_f16(g_bufs["raw_g_2"].d, g_bufs["raw_g_2h"].d, 1*2*83443*117L, s);
    cast_f32_f16(g_bufs["dc_s_2"].d, g_bufs["dc_s_2h"].d, 1*2*83443*32L, s);
    ada_launch_c(g_bufs["raw_g_2h"].d, g_bufs["dc_s_2h"].d, g_bufs["xy_s2"].d,
                 near_h, far_h, extr_h, extr_h, 0,
                 (float*)g_bufs["final_means"].d + 52150L*3,
                 (float*)g_bufs["final_cov"].d + 52150L*9,
                 (float*)g_bufs["final_harm"].d + 52150L*75,
                 (float*)g_bufs["final_opa"].d + 52150L,
                 2, 83443, 0.5f, 15.f, 256.f, 512.f, s);
    segend(27);
    seg(28);
    if (!eng("gh_s3_cnn").ctx->enqueueV3(s)) fprintf(stderr, "ENQUEUE FAIL\n");
    segend(28);
    seg(29);
    if (!eng("gh_s3_mlp").ctx->enqueueV3(s)) fprintf(stderr, "ENQUEUE FAIL\n");
    cast_f32_f16(g_bufs["raw_g_3"].d, g_bufs["raw_g_3h"].d, 1*2*333772*117L, s);
    cast_f32_f16(g_bufs["dc_s_3"].d, g_bufs["dc_s_3h"].d, 1*2*333772*32L, s);
    ada_launch_c(g_bufs["raw_g_3h"].d, g_bufs["dc_s_3h"].d, g_bufs["xy_s3"].d,
                 near_h, far_h, extr_h, extr_h, 0,
                 (float*)g_bufs["final_means"].d + 219036L*3,
                 (float*)g_bufs["final_cov"].d + 219036L*9,
                 (float*)g_bufs["final_harm"].d + 219036L*75,
                 (float*)g_bufs["final_opa"].d + 219036L,
                 2, 333772, 0.5f, 15.f, 512.f, 1024.f, s);
    segend(29);
    seg(30);
    float* fm = (float*)g_bufs["final_means"].d;
    float* fc = (float*)g_bufs["final_cov"].d;
    float* fh = (float*)g_bufs["final_harm"].d;
    float* fo = (float*)g_bufs["final_opa"].d;
    CK(cudaMemcpyAsync(fm, g_bufs["means_0"].d, 10430L*3*4, cudaMemcpyDeviceToDevice, s));
    CK(cudaMemcpyAsync(fc, g_bufs["covariances_0"].d, 10430L*9*4, cudaMemcpyDeviceToDevice, s));
    CK(cudaMemcpyAsync(fh, g_bufs["harmonics_0"].d, 10430L*75*4, cudaMemcpyDeviceToDevice, s));
    CK(cudaMemcpyAsync(fo, g_bufs["opacities_0"].d, 10430L*4, cudaMemcpyDeviceToDevice, s));
    CK(cudaMemcpyAsync(fm + 10430L*3, g_bufs["means_1"].d, 41720L*3*4, cudaMemcpyDeviceToDevice, s));
    CK(cudaMemcpyAsync(fc + 10430L*9, g_bufs["covariances_1"].d, 41720L*9*4, cudaMemcpyDeviceToDevice, s));
    CK(cudaMemcpyAsync(fh + 10430L*75, g_bufs["harmonics_1"].d, 41720L*75*4, cudaMemcpyDeviceToDevice, s));
    CK(cudaMemcpyAsync(fo + 10430L, g_bufs["opacities_1"].d, 41720L*4, cudaMemcpyDeviceToDevice, s));
    segend(30);
  };


  // warmup
  for (int it = 0; it < 3; ++it) chain_once(false);
  CK(cudaStreamSynchronize(s));

  // timed iterations (PANSPLAT_RUNNER_ITERS overrides; a longer run gives a
  // steadier min — the Orin contract numbers use min-of-50 in a silent window)
  int ITERS = 10;
  if (const char* e = getenv("PANSPLAT_RUNNER_ITERS")) {
    int v = atoi(e);
    if (v > 0) ITERS = v;
  }
  double acc[NS] = {0}; double best[NS] = {0};
  double accT = 0, bestT = 1e9;
  for (int it = 0; it < ITERS; ++it) {
    chain_once(true);
    CK(cudaStreamSynchronize(s));
    double tot = 0;
    for (int i = 0; it == 0 && i < NS; ++i) {}
    for (int i = 0; i < NS; ++i) {
      double v = ms_of(ev0[i], ev1[i]);
      acc[i] += v;
      if (it == 0 || v < best[i]) best[i] = v;
      tot += v;
    }
    accT += tot;
    if (tot < bestT) bestT = tot;
  }
  const char* names[NS] = {"mvs_pre", "cv_pre0", "cvk0", "cv_post0",
                           "cv_pre1", "cvk1",
                           "cv_post1a", "fl1a", "cv_post1m1", "fl1b",
                           "cv_post1m2", "fl1c", "cv_post1b",
                           "cv_pre2", "cvk2",
                           "cv_post2a", "fl2a", "cv_post2m1", "fl2b",
                           "cv_post2m2", "fl2c", "cv_post2b",
                           "s0_cnn", "s0_mlp+ada", "s1_cnn", "s1_mlp+ada",
                           "s2_cnn", "s2_mlp+ada", "s3_cnn", "s3_mlp+ada", "concat"};
  printf("\n=== per-segment latency (ms, min/avg of %d iters) ===\n", ITERS);
  for (int i = 0; i < NS; ++i)
    printf("%-16s %8.2f / %8.2f\n", names[i], best[i], acc[i] / ITERS);
  printf("%-16s %8.2f / %8.2f\n", "TOTAL", bestT, accT / ITERS);
  // ---- dump final gaussians ----
  CK(cudaStreamSynchronize(s));
  dump_dev(g_bufs["final_means"].d, 886580L*3*4, (std::string(out_path) + ".means.f32").c_str());
  dump_dev(g_bufs["final_cov"].d, 886580L*9*4, (std::string(out_path) + ".cov.f32").c_str());
  dump_dev(g_bufs["final_harm"].d, 886580L*75*4, (std::string(out_path) + ".harm.f32").c_str());
  dump_dev(g_bufs["final_opa"].d, 886580L*4, (std::string(out_path) + ".opa.f32").c_str());
  printf("gaussians dumped\n");
  // Destroy engines/contexts while the CUDA driver is still alive: letting
  // static destruction handle them fires every TRT teardown path after the
  // driver has shut down (error spam, and a segfault at exit on mixed TRT
  // installs). Nothing here is reused, so the process can end right after.
  g_engs.clear();
  return 0;
}
