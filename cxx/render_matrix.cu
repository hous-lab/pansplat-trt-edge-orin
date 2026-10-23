// Torch-free render matrix bench: parameterized #faces x per-face resolution.
// Extended copy of cxx/render_bench.cu (original file untouched).
//
// Usage:
//   render_matrix <data_dir> [--faces=1,2,6] [--res=128,256,512,1024] [--iters=20]
//     --faces list: how many views to render per timed iteration
//     --res list:   per-face resolution (square), FOV held constant (tanx/tany from faces.bin)
//   Data dir: export/render_bench layout (means/cov6/opa/shs .f32 + faces.bin + meta.json)
//
// Build:
//   nvcc -O3 -std=c++17 -arch=$ARCH -I$DGR/cuda_rasterizer -I$DGR/third_party/glm \
//     cxx/render_matrix.cu $DGR/cuda_rasterizer/{forward,backward,rasterizer_impl}.cu -o render_matrix
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <functional>
#include <fstream>
#include <cassert>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include "rasterizer.h"
using namespace CudaRasterizer;

#define CK(x) do { auto e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)

static std::vector<char> read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  assert(f.good());
  f.seekg(0, std::ios::end);
  size_t n = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<char> v(n);
  f.read(v.data(), n);
  return v;
}

// parse "1,2,6" -> {1,2,6}; returns count
static int parse_list(const char* s, int* out, int maxn) {
  int n = 0; const char* p = s;
  while (*p && n < maxn) {
    out[n++] = (int)strtol(p, (char**)&p, 10);
    if (*p == ',') ++p;
  }
  return n;
}

int main(int argc, char** argv) {
  const char* dir = argc > 1 ? argv[1] : "export/render_bench";
  const char* rs = std::getenv("DGR_SRC");
  assert(rs);
  int faces_list[8] = {1, 2, 6};
  int nfaces_list = 3;
  int res_list[8] = {128, 256, 512, 1024};
  int nres_list = 4;
  int iters = std::getenv("ITERS") ? atoi(std::getenv("ITERS")) : 20;
  int warmup = 3;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--faces=", 0) == 0) nfaces_list = parse_list(a.c_str() + 8, faces_list, 8);
    else if (a.rfind("--res=", 0) == 0) nres_list = parse_list(a.c_str() + 6, res_list, 8);
    else if (a.rfind("--iters=", 0) == 0) iters = atoi(a.c_str() + 8);
  }
  cudaFree(0);
  // ---- load gaussians ----
  auto means_v = read_file(std::string(dir) + "/means.f32");
  auto cov_v = read_file(std::string(dir) + "/cov6.f32");
  auto opa_v = read_file(std::string(dir) + "/opa.f32");
  auto shs_v = read_file(std::string(dir) + "/shs.f32");
  auto faces_v = read_file(std::string(dir) + "/faces.bin");
  int P = 886580;
  {
    std::ifstream mf(std::string(dir) + "/meta.json");
    std::string s((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
    auto pos = s.find("\"P\": ") + 5;
    P = atoi(s.substr(pos, s.substr(pos).find(",")).c_str());
  }
  printf("P=%d iters=%d\n", P, iters);
  float *d_means, *d_cov, *d_opa, *d_shs;
  CK(cudaMalloc(&d_means, means_v.size()));
  CK(cudaMalloc(&d_cov, cov_v.size()));
  CK(cudaMalloc(&d_opa, opa_v.size()));
  CK(cudaMalloc(&d_shs, shs_v.size()));
  const float* fp = (const float*)faces_v.data();
  // per-face record: viewmat[16] projmat[16] campos[3] tanx tany bg[3] = 40 floats
  const int FACE_FLOATS = 40;
  std::vector<std::vector<float>> fh(6);
  for (int i = 0; i < 6; ++i) {
    const float* rec = fp + (size_t)i * FACE_FLOATS;
    fh[i].assign(rec, rec + FACE_FLOATS);
  }
  // scale-invariant normalization (decoder does this before projection)
  float near0 = 0.44995117f;
  float sc = 1.f / near0;
  if (getenv("NO_SCALE")) sc = 1.f;
  {
    std::vector<float> m((float*)means_v.data(), (float*)means_v.data() + P * 3);
    std::vector<float> c((float*)cov_v.data(), (float*)cov_v.data() + P * 6);
    for (long i = 0; i < (long)P * 3; ++i) m[i] *= sc;
    for (long i = 0; i < (long)P * 6; ++i) c[i] *= sc * sc;
    CK(cudaMemcpy(d_means, m.data(), P * 3 * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_cov, c.data(), P * 6 * 4, cudaMemcpyHostToDevice));
  }
  CK(cudaMemcpy(d_opa, opa_v.data(), opa_v.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_shs, shs_v.data(), shs_v.size(), cudaMemcpyHostToDevice));
  float *d_vm, *d_pm, *d_cp, *d_bg, *d_out, *d_radii;
  CK(cudaMalloc(&d_vm, 16 * 4));
  CK(cudaMalloc(&d_pm, 16 * 4));
  CK(cudaMalloc(&d_cp, 3 * 4));
  CK(cudaMalloc(&d_bg, 3 * 4));
  CK(cudaMalloc(&d_out, 3 * 1024 * 1024 * 4));  // max res 1024^2 x 3ch x f32
  CK(cudaMalloc(&d_radii, P * 4));
  float* d_shs_nc = d_shs;
  float* d_cov_nc = d_cov;
  float scale_modifier = 1.f;
  float tanx = fh[0][32], tany = fh[0][33];
  // geometry/binning/image buffers (resizable; growth outside timed region via warmup)
  size_t gsz = 0, bsz = 0, isz = 0;
  char *gbuf = nullptr, *bbuf = nullptr, *ibuf = nullptr;
  std::function<char*(size_t)> gb = [&](size_t n) {
    if (n > gsz) { if (gbuf) CK(cudaFree(gbuf)); CK(cudaMalloc(&gbuf, n)); gsz = n; }
    return gbuf;
  };
  std::function<char*(size_t)> bb = [&](size_t n) {
    if (n > bsz) { if (bbuf) CK(cudaFree(bbuf)); CK(cudaMalloc(&bbuf, n)); bsz = n; }
    return bbuf;
  };
  std::function<char*(size_t)> ib = [&](size_t n) {
    if (n > isz) { if (ibuf) CK(cudaFree(ibuf)); CK(cudaMalloc(&ibuf, n)); isz = n; }
    return ibuf;
  };
  int D = 3, M = 25;
  const bool use_dc = getenv("USE_DC") != nullptr;
  float* d_rgb = nullptr;
  if (use_dc) {
    std::vector<float> rgb(P * 3);
    const float* sh = (const float*)shs_v.data();
    const float C0 = 0.28209479177387814f;
    for (int i = 0; i < P; ++i)
      for (int c = 0; c < 3; ++c) {
        float v = 0.5f + C0 * sh[i * 75 + c];
        rgb[i * 3 + c] = v < 0 ? 0 : v;
      }
    CK(cudaMalloc(&d_rgb, P * 3 * 4));
    CK(cudaMemcpy(d_rgb, rgb.data(), P * 3 * 4, cudaMemcpyHostToDevice));
    D = 0;
    M = 1;
  }
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int fi = 0; fi < nfaces_list; ++fi) {
    for (int ri = 0; ri < nres_list; ++ri) {
      int nf = faces_list[fi];
      int R = res_list[ri];
      int nrendered_acc = 0;
      auto render_face = [&](int i) {
        const std::vector<float>& r = fh[i];
        CK(cudaMemcpy(d_vm, r.data(), 16 * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_pm, r.data() + 16, 16 * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_cp, r.data() + 32, 3 * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_bg, r.data() + 37, 3 * 4, cudaMemcpyHostToDevice));
        tanx = r[35];
        tany = r[36];
        int n_rad = Rasterizer::forward(gb, bb, ib, P, D, M, d_bg, R, R,
                            d_means, use_dc ? nullptr : d_shs, use_dc ? d_rgb : nullptr, d_opa, nullptr,
                            scale_modifier, nullptr, d_cov_nc,
                            d_vm, d_pm, d_cp, tanx, tany, false,
                            d_out, (int*)d_radii, false);
        return n_rad;
      };
      auto do_render = [&]() {
        for (int i = 0; i < nf; ++i) nrendered_acc = render_face(i);
      };
      for (int w = 0; w < warmup; ++w) do_render();
      CK(cudaDeviceSynchronize());
      float best = 1e9f, acc = 0.f;
      for (int it = 0; it < iters; ++it) {
        CK(cudaEventRecord(e0));
        do_render();
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        best = best < ms ? best : ms;
        acc += ms;
      }
      double pix = (double)nf * R * R;
      printf("faces=%d res=%4d  min=%8.3f avg=%8.3f  nrendered=%7d  %.2f Mpixels  %.2f ns/pix  %.1f fps\n",
             nf, R, best, acc / iters, nrendered_acc, pix / 1e6,
             1e3 * acc / iters / pix * 1e6, 1000.0 * iters / acc);
      fflush(stdout);
    }
  }
  printf("done\n");
  return 0;
}
