# 02 — Pipeline

`run_pipeline.sh` drives the full deployment chain in 8 idempotent steps. Each step writes a sentinel under `export/.ok/`; delete a sentinel to force rework of that step. Re-runs skip completed engines (per-engine files), so a failed step 4 recovers without rebuilding everything.

## Tiers

| Tier | Invocation | Inputs | Gate |
|---|---|---|---|
| golden | `PANSPLAT_INPUTS=golden bash run_pipeline.sh` | preprocessed MP3D scene via the upstream dataloader | 4-field cosine + render PSNR vs `export/golden_512.npz` |
| synth | `bash run_pipeline.sh` (default) | seeded-random 10-bin synthetic inputs, shapes locked to the runner contract | self-anchored gate: runner dump vs torch chain on identical inputs, cos ≥ 0.9997 |
| HF | `PANSPLAT_SKIP_EXPORT=1 bash run_pipeline.sh` | ONNX fetched from the HF repo | validates 21 ONNX against the manifest |

Synthetic tier exists so the whole chain can be exercised without any dataset: `scripts/make_synth_inputs.py` writes the 10 binaries (image / mono_image / cube_image / extrinsics / near / far / xy_s0-s3) with shapes locked to runner.cu's buffer table; `scripts/verify_synth_gate.py` runs the torch reference chain on the same inputs (overriding the gaussian head's `gs_xy_*_0` buffers from the same bins) and compares per-field cosine. Deployment equivalence, not dataset accuracy.

## The 8 steps

1. **guard** — manifest `--check` (drift detection against `cxx/runner.cu`), checkpoint presence, trtexec on PATH, torch 2.4.0 pin
2. **runner inputs** — golden npz (torch trace scene, produced by `scripts/dump_golden.py` if absent) + golden or synth 10-bin runner inputs
3. **ONNX export** — `scripts/export_all.py` traces 23 graphs (21 final + cv_post1/2 intermediates); `PANSPLAT_SKIP_EXPORT=1` instead validates HF-fetched ONNX against the manifest
4. **engines** — `detect_platform.py` keys engines by arch+TRT version into `export/engines/<platform>/`; trtexec FP16 builds with per-engine logs; ~41 GB free RAM needed for the gh_s3 pair (Myelin workspace), hard guard
5. **build cxx** — `make -C cxx all` (single nvcc, no torch dependency)
6. **accuracy gates** — golden tier: runner dump → `compare_trt_golden.py` (4-field cos + render PSNR); synth tier: `verify_synth_gate.py`
7. **latency bench** — Orin: MAXN + quiet window, min-of-N discipline (docs/03); `PANSPLAT_RUNNER_ITERS` raises the timed-iteration count (default 10; a higher N gives a steadier min, e.g. 50); x86: chain timing + per-engine numbers via `scripts/bench_trt.py`
8. **summary** — contract reference numbers banner

## Platform note

`scripts/detect_platform.py` names the engine directory by architecture + TensorRT version (e.g. `x86_64-trt10.3`), so engines built for one TRT pairing are never silently reused on another. A6000 (sm_86, TRT 10.16) and Orin (sm_87, TRT 10.3) get separate directories. Add your platform's engines under `export/engines/<platform>/` (or let step 4 build them) and the rest of the chain is platform-agnostic.
