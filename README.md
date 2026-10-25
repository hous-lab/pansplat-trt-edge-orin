# PanSplat TRT Edge — Jetson AGX Orin Deployment

Real-time feed-forward panoramic 3D Gaussian Splatting on Jetson AGX Orin.

**Status: Active.** Results last verified 2026-10 on Jetson AGX Orin 64GB (JetPack 6.2, TensorRT 10.3), RTX A6000 (TensorRT 10.16) and a 4090 dev host (TensorRT 10.4, equivalence cross-checks). Issues and PRs welcome.

[PanSplat](https://github.com/chengzhag/PanSplat) (CVPR 2025) reconstructs a 3D Gaussian scene from two 512×1024 ERP images in a single forward pass. The stock PyTorch pipeline takes ~30 s per frame on Orin — this project deploys it as **21 TensorRT FP16 engines plus 5 custom CUDA kernel classes (41 kernel launches) forming a 31-segment GPU-resident chain**, and cuts end-to-end latency to **~567 ms** with no measurable quality loss.

## Results

Two latency calibers are reported separately and never mixed: the chain caliber (model-inference only) and the end-to-end caliber (chain + rendering). Full measurement contract: [docs/03_benchmark_contract.md](docs/03_benchmark_contract.md).

**Chain latency (model-inference)** - Jetson AGX Orin 64GB, JetPack 6.2, TensorRT 10.3, MAXN + live clock readback (GR3D 1300.5 MHz), quiet window, 3 untimed warmup iterations then min/mean of N≥5 (2026-10 release re-run: N=50):

| Metric | Value |
|---|---|
| Reconstruction (2×512×1024 ERP → 886,580 Gaussians × 88 dims) | **546.3–551.2 ms** (min / mean) |
| 2026-10 release re-run (quiet cool window, 99% util) | 555.1 ms min / 556.4 ms mean (N=50), repeat 555.1–561.0 |
| Rendering (886,580 Gaussians → 6×256² cubemap → 512×1024 ERP) | 20.6 ms |
| RTX A6000 recon (TRT 10.16, shared card) | 129.97 ms (±10%) |

**End-to-end & platform overhead**:

| Metric | Value |
|---|---|
| End-to-end (chain + render summed) | **~567 ms** |
| vs PyTorch eager on the same Orin (30,000 ms) | 53.7× |
| H2D/D2H (measured separately, excluded from e2e) | 6–13 ms |
| Cold start (to first frame, separate from steady state) | 3.1–3.6 s |

**Accuracy gates (golden-anchor tier)**:

| Metric | Value |
|---|---|
| Gaussian fields cosine vs PyTorch golden (means/cov/harmonics/opacity) | ≥ 0.99978 per field |
| Render PSNR vs PyTorch golden render (same-env A/B) | 52.95 dB (golden 52.96 dB) |
| 2026-10 clean-room golden gate (A6000 / 4090 dev) | 4-field cosine PASS, render PSNR 53.22 / 53.15 dB (floor 52.0) |

## Architecture

The 2-view encoder (MVS backbone + cost volumes + 4-stage gaussian head) traces to 23 ONNX graphs; 21 engines are loaded at runtime:

- 1× `mvs_pre` (backbone + mono fusion)
- 7× cost-volume cascade (`cv_pre0-2` + CV-kernel inline + `cv_post0-2` split)
- 8× 4-stage gaussian head (`gh_s{0-3}_{cnn,mlp}`)
- 4+4× attention splits (`p1_`/`p2_`) around two custom flash-attention kernels (H=2/S=4096/D=32 and H=1/S=16384/D=32)

The 5 custom kernel classes replace graph segments where TensorRT's Myelin compiler systematically fails (see [docs/04_kernels.md](docs/04_kernels.md)): cost volume (70× off the bandwidth roofline as fused graph), adapter chain (fp32 K=3 GEMMs on FFMA+trig chains, 32× speedup), flash attention (avoid materializing 16384² fp16 scores), boundary dtype casts, and a torch-free render bench (direct-call of the forked rasterizer, 20.6 ms).

Engine-internal math stays FP32; engines take FP16 inputs. Segment table, equivalence method, and the full audit trail: [docs/01_architecture.md](docs/01_architecture.md), [docs/05_accuracy.md](docs/05_accuracy.md), [docs/08_results.md](docs/08_results.md).

## Quickstart

```bash
# 1. Environment (conda env from environment_strict.yaml; python 3.10, torch 2.4.0+cu118, TRT 10.3)
#    trtexec must be on PATH (engines are built with the TRT CLI only -
#    the pip python API segfaults on these graphs).

# 2. Checkpoint: download per ckpts/README.md into
#    logs/nvpl49ge/checkpoints/last.ckpt   (sha256 in ckpts/README.md)

# 3. Golden tier = the accuracy gate (needs the golden anchor npz -
#    dataset-free; fetch from the HF artifacts repo: https://huggingface.co/hous-lab/pansplat-trt-edge-orin)
#    - see export/README.md
PANSPLAT_INPUTS=golden bash run_pipeline.sh

# 4. Smoke tier (default): pipeline bring-up only. The 21-engine chain
#    bakes the golden-scene encoding - see docs/05_accuracy.md - so a
#    non-golden input set has no numerical meaning here.
bash run_pipeline.sh
```

The pipeline is 8 idempotent steps with `export/.ok/` sentinels: guard → runner inputs → ONNX export (23 graphs) → 21 trtexec engine builds → CUDA runner build → accuracy gates → latency bench → summary. `PANSPLAT_SKIP_EXPORT=1` validates HF-fetched ONNX against the manifest and skips tracing — see [docs/02_pipeline.md](docs/02_pipeline.md).

## Repository Map

- [cxx/](cxx) — GPU-resident runner (~40 KB single-file C++/CUDA) + 5 kernel classes + `Makefile` (single nvcc, no torch dependency)
- [scripts/](scripts) — export chain, manifest tooling, gates, benches
- [src/](src) — deployment-side model surgery (segment wrappers: [mvs_split.py](src/model/encoder/mvs_split.py), [gh_split.py](src/model/encoder/gh_split.py), Rodrigues real SH rotation, SDPA backend, SE(3) closed-form inverse)
- [export/](export) — regenerable artifacts (ONNX, engines, runner inputs, gates); see [export/README.md](export/README.md)
- [third_party/diff-gaussian-rasterization-modified/](third_party/diff-gaussian-rasterization-modified) — forked rasterizer (vendored, Inria/MPII non-commercial research license - see [NOTICE](NOTICE) and [third_party/VENDORED.md](third_party/VENDORED.md))

## Scope & limitations

- Single-scene feature-cache deployment: the gh_s*_cnn engines bake the golden-scene encoding, so numerical gates run on the golden-anchor tier ([docs/05_accuracy.md](docs/05_accuracy.md)); non-golden inputs are smoke-tier only.
- Not solved here: INT8 (three documented strikes, [docs/08_results.md](docs/08_results.md)), cold start (3.1–3.6 s, untreated), interactive viewing (the render bench is a benchmark binary, not a viewer), DLA.
- Conclusions are bound to the platform pairings in [docs/07_deployment.md](docs/07_deployment.md); engines are not portable across TRT versions or GPU architectures.
- The 256-res tier latency was measured (A6000 ~79 ms) but quality was never evaluated - decision data missing, tier frozen.

## License

MIT for this deployment code; PanSplat model code © Cheng Zhang, MIT; the vendored rasterizer fork carries its own license — see [NOTICE](NOTICE). The PanSplat checkpoints remain under their original license and are link-only (see [ckpts/README.md](ckpts/README.md)).
