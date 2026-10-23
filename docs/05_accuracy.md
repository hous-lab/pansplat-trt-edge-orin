# 05 — Accuracy & equivalence method

The chain must produce the same Gaussians as the torch model. Equivalence is verified at two tiers, both re-runnable from the pipeline.

## Golden tier

`scripts/dump_golden.py` runs the torch model (upstream weights, predict mode) on a preprocessed MP3D scene and dumps `export/golden_512.npz`: context tensors + the four Gaussian fields from the torch model. This npz is the trace scene for the whole export chain AND the reference for the golden gate.

After the runner chain completes on the same inputs (`export/runner_bins/`), `scripts/compare_trt_golden.py` compares the four Gaussian fields (means / covariances / harmonics / opacities): per-field cosine ≥ 0.99978 and renders the TRT Gaussians through the vendored rasterizer for a render PSNR vs the torch render (52.95 vs 52.96 dB — lossless).

`scripts/verify_chain_gaussians.py` is the fast sanity variant (cosine only, CHAIN-EQUIV-PASS/FAIL exit code).

## Synth tier (dataset-free)

`scripts/make_synth_inputs.py` writes seeded-random inputs with shapes locked to the runner contract (10 bins). `scripts/verify_synth_gate.py` runs the torch reference chain on the identical inputs — overriding the gaussian head's `gs_xy_*_0` buffers from the same bins — and compares per-field cosine ≥ 0.9997 (deployment equivalence, not dataset accuracy).

This tier exists because the golden npz derives from MP3D pixels: HF users can run the synth tier end-to-end without any dataset. The synth golden npz + synth bins on the HF repo are the same self-anchored gate, pre-runnable without torch.

## What the gate does not claim

- The gate is equivalence, not model accuracy — no dataset eval is bundled (single golden scene).
- Thresholds: golden 0.99978 / synth 0.9997 are the measured deployment equivalence on this code state; they are not statistical guarantees.
- Render PSNR is same-environment A/B only.
