# export/ — regenerable deployment artifacts

Everything under `export/` is **regenerable and gitignored**. A fresh clone
ships this directory empty (this README aside); `run_pipeline.sh` populates it:

- `export/onnx/` — 21 segment ONNX graphs produced by the export chain
  (scripts in `scripts/`, orchestrated by step 3 of `run_pipeline.sh`).
- `export/engines/<platform>/` — TensorRT FP16 engines built per platform
  (step 4). `<platform>` is `orin-trt10.3` or `x86-trt10.16` (auto-detected).
- `export/runner_bins/` — runner input binaries (10), local runtime
  artifacts, never committed. Golden tier: rebuild with
  `scripts/prepare_runner_inputs.py` from the anchor npz + checkpoint xy
  grids (dataset-free). Smoke tier: `scripts/make_synth_inputs.py`
  (random context + checkpoint xy grids).
- `export/golden_512.npz` — golden anchor (4 gaussian-parameter tensors of the
  reference scene from the PyTorch model, used by the accuracy gates).
  Self-produced; published on the Hugging Face companion repo. It is
  the accuracy anchor AND the golden-tier input source - consumers need
  it for the numerical gate. Regenerable via `scripts/dump_golden.py` from your own
  checkpoint + dataset.
- `export/.ok/` — content-sha sentinels used by `run_pipeline.sh` for
  idempotency (safe to delete at any time; deletion only forces rework).

Rebuild everything from scratch:

```bash
bash run_pipeline.sh            # all steps, idempotent
PANSPLAT_SKIP_EXPORT=1 bash run_pipeline.sh   # skip ONNX export (use fetched ONNX)
```
