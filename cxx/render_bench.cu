// Torch-free render bench: calls CudaRasterizer::Rasterizer::forward directly.
// Build (Orin): nvcc -O3 -std=c++17 -arch=sm_87 -I<repo>/cuda_rasterizer -I<repo>/third_party/glm \
//   cxx/render_bench.cu <repo>/cuda_rasterizer/*.cu -o render_bench
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

int main(int argc, char** argv) {
  const char* dir = argc > 1 ? argv[1] : "export/render_bench";
  const char* rs = argc > 2 ? argv[2] : std::getenv("DGR_SRC");
  assert(rs);
  std::string src = rs;
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
  printf("P=%d faces=6 face_w=256\n", P);
  float *d_means, *d_cov, *d_opa, *d_shs;
  CK(cudaMalloc(&d_means, means_v.size()));
  CK(cudaMalloc(&d_cov, cov_v.size()));
  CK(cudaMalloc(&d_opa, opa_v.size()));
  CK(cudaMalloc(&d_shs, shs_v.size()));
  CK(cudaMemcpy(d_means, means_v.data(), means_v.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_cov, cov_v.data(), cov_v.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_opa, opa_v.data(), opa_v.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_shs, shs_v.data(), shs_v.size(), cudaMemcpyHostToDevice));
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
  CK(cudaMemcpy(d_means, means_v.data(), means_v.size(), cudaMemcpyHostToDevice));
  {
    // apply scale on host copy then upload
    std::vector<float> m((float*)means_v.data(), (float*)means_v.data() + P * 3);
    std::vector<float> c((float*)cov_v.data(), (float*)cov_v.data() + P * 6);
    for (long i = 0; i < (long)P * 3; ++i) m[i] *= sc;
    for (long i = 0; i < (long)P * 6; ++i) c[i] *= sc * sc;
    CK(cudaMemcpy(d_means, m.data(), P * 3 * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_cov, c.data(), P * 6 * 4, cudaMemcpyHostToDevice));
  }
  float *d_vm, *d_pm, *d_cp, *d_bg, *d_out, *d_radii;
  CK(cudaMalloc(&d_vm, 16 * 4));
  CK(cudaMalloc(&d_pm, 16 * 4));
  CK(cudaMalloc(&d_cp, 3 * 4));
  CK(cudaMalloc(&d_bg, 3 * 4));
  CK(cudaMalloc(&d_out, 3 * 256 * 256 * 4));
  CK(cudaMalloc(&d_radii, P * 4));
  float* d_shs_nc = d_shs;
  float* d_cov_nc = d_cov;
  float scale_modifier = 1.f;
  float tanx = fh[0][32], tany = fh[0][33];
  // geometry/binning/image buffers via resizable lambdas
  size_t gsz = 0, bsz = 0, isz = 0;
  char *gbuf = nullptr, *bbuf = nullptr, *ibuf = nullptr;
  std::function<char*(size_t)> gb = [&](size_t n) {
    if (n > gsz) { if (gbuf) CK(cudaFree(gbuf)); CK(cudaMalloc(&gbuf, n)); gsz = n; }
    return gbuf;
  };
  std::function<char*(size_t)> bb = [&](size_t n) {
    if (n > bsz) {
      if (bbuf) CK(cudaFree(bbuf));
      CK(cudaMalloc(&bbuf, n));
      bsz = n;
    }
    return bbuf;
  };
  std::function<char*(size_t)> ib = [&](size_t n) {
    if (n > isz) {
      if (ibuf) CK(cudaFree(ibuf));
      CK(cudaMalloc(&ibuf, n));
      isz = n;
    }
    return ibuf;
  };
  int D = 3, M = 25;
  float* d_rgb = nullptr;
  const bool use_dc = getenv("USE_DC") != nullptr;
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
  int nrendered[6] = {0};
  auto render_face = [&](int i) {
    const std::vector<float>& r = fh[i];
      CK(cudaMemcpy(d_vm, r.data(), 16 * 4, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(d_pm, r.data() + 16, 16 * 4, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(d_cp, r.data() + 32, 3 * 4, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(d_bg, r.data() + 37, 3 * 4, cudaMemcpyHostToDevice));
      tanx = r[35];
      tany = r[36];
      int n_rad = 0;
      n_rad = Rasterizer::forward(gb, bb, ib, P, D, M, d_bg, 256, 256,
                          d_means, use_dc ? nullptr : d_shs, use_dc ? d_rgb : nullptr, d_opa, nullptr,
                          scale_modifier, nullptr, d_cov_nc,
                          d_vm, d_pm, d_cp, tanx, tany, false,
                          d_out, (int*)d_radii, false);
      return n_rad;
  };
  auto do_render = [&]() {
    for (int i = 0; i < 6; ++i) nrendered[i] = render_face(i);
  };
  do_render();
  CK(cudaDeviceSynchronize());
  printf("num_rendered per face: ");
  for (int i = 0; i < 6; ++i) printf("%d ", nrendered[i]);
  printf("\n");
  {
    std::vector<int> rad(P);
    CK(cudaMemcpy(rad.data(), d_radii, P * 4, cudaMemcpyDeviceToHost));
    int vis = 0;
    for (int i = 0; i < P; ++i) vis += rad[i] > 0;
    printf("visible gaussians (radii>0, last face): %d / %d\n", vis, P);
  }
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  const int IT = getenv("ITERS") ? atoi(getenv("ITERS")) : 10;
  float best = 1e9f, acc = 0.f;
  for (int it = 0; it < IT; ++it) {
    CK(cudaEventRecord(e0));
    do_render();
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    best = best < ms ? best : ms;
    acc += ms;
  }
  printf("render 6-face: best=%.2fms avg=%.2fms\n", best, acc / IT);
  std::vector<float> host(3 * 256 * 256);
  for (int i = 0; i < 6; ++i) {
    render_face(i);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(host.data(), d_out, host.size() * 4, cudaMemcpyDeviceToHost));
    double s = 0;
    float mn = 1e9f, mx = -1e9f;
    for (float v : host) { s += v; mn = v < mn ? v : mn; mx = v > mx ? v : mx; }
    char p[96];
    snprintf(p, sizeof(p), "export/scratch/render_face%d.f32", i);
    FILE* f2 = fopen(p, "wb");
    fwrite(host.data(), 4, host.size(), f2);
    fclose(f2);
    printf("face %d: mean %.4f [%.4f,%.4f]\n", i, s / host.size(), mn, mx);
  }
  printf("done\n");
  return 0;
}
