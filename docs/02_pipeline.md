# 02 — Pipeline

`run_pipeline.sh` drives the full deployment chain in 8 idempotent steps. Each step writes a sentinel under `export/.ok/`; delete a sentinel to force rework of that step. Re-runs skip completed engines (per-engine files), so a failed step 4 recovers without rebuilding everything.

## Tiers

| Tier | Invocation | Inputs | Gate |
|---|---|---|---|
| golden | `PANSPLAT_INPUTS=golden bash run_pipeline.sh` | preprocessed MP3D scene via the upstream dataloader | 4-field cosine + render PSNR vs `export/golden_512.npz` |
| smoke | `bash run_pipeline.sh` (default) | seeded-random context + checkpoint xy grids (10 bins) | structural gate only (`verify_smoke.py`: dump sizes + finiteness) - NOT a numerical gate |
| HF | `PANSPLAT_SKIP_EXPORT=1 bash run_pipeline.sh` | ONNX fetched from the HF repo | validates 21 ONNX against the manifest |

The smoke tier exercises the whole chain without any dataset: `scripts/make_synth_inputs.py` writes the 10 binaries (random context + checkpoint xy grids, shapes locked to runner.cu's buffer table). It is a bring-up / integrity check, not a numerical gate: the gh_s*_cnn engines carry the golden scene's encoded features as graph constants (runner.cu binds no features/correlation/image to them; gh_s0_cnn takes no inputs at all), and the gh_s*_mlp engines bake the xy grids. The deployment is a single-scene feature-cache design (docs/01_architecture.md); with non-golden inputs the outputs are not numerically meaningful.

## The 8 steps

1. **guard** — manifest `--check` (drift detection against `cxx/runner.cu`), checkpoint presence, trtexec on PATH, torch 2.4.0 pin
2. **runner inputs** — golden: anchor npz (HF download, or dump_golden.py from a preprocessed MP3D scene) + checkpoint xy grids → 10 bins; smoke: make_synth_inputs.py (random ctx + ckpt xy)
3. **ONNX export** — `scripts/export_all.py` traces 23 graphs (21 final + cv_post1/2 intermediates); `PANSPLAT_SKIP_EXPORT=1` instead validates HF-fetched ONNX against the manifest
4. **engines** — `detect_platform.py` keys engines by arch+TRT version into `export/engines/<platform>/`; trtexec FP16 builds with per-engine logs; ~41 GB free RAM needed for the gh_s3 pair (Myelin workspace), hard guard
5. **build cxx** — `make -C cxx all` (single nvcc, no torch dependency)
6. **accuracy gates** — golden tier: runner dump → `compare_trt_golden.py` (4-field cos ≥ 0.99978 + render PSNR, gate verdict drives the exit code); smoke tier: `verify_smoke.py` (structure only)
7. **latency bench** — Orin: MAXN + quiet window, min-of-N discipline (docs/03); `PANSPLAT_RUNNER_ITERS` raises the timed-iteration count (default 10; a higher N gives a steadier min, e.g. 50); x86: chain timing + per-engine numbers via `scripts/bench_trt.py`
8. **summary** — contract reference numbers banner

## Platform note

`scripts/detect_platform.py` names the engine directory by architecture + TensorRT version (e.g. `x86_64-trt10.3`), so engines built for one TRT pairing are never silently reused on another. A6000 (sm_86, TRT 10.16) and Orin (sm_87, TRT 10.3) get separate directories. Add your platform's engines under `export/engines/<platform>/` (or let step 4 build them) and the rest of the chain is platform-agnostic.
