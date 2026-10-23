# 05 — Accuracy & equivalence method

The chain must produce the same Gaussians as the torch model on the deployed
scene. Equivalence is verified on the golden-anchor tier, re-runnable from
the pipeline. The deployment itself is a single-scene feature-cache design
(see docs/01_architecture.md): the gh_s*_cnn engines carry the golden
scene's encoded features as graph constants, and the gh_s*_mlp engines bake
the xy sampling grids from the checkpoint. Only the depth hypotheses
(disp_candi, from the cost-volume path) and the residual chain (last_raw)
vary at runtime.

## Golden-anchor tier (the accuracy gate)

`export/golden_512.npz` is the self-produced anchor pack: the golden
scene's context tensors plus the four Gaussian fields from the torch
reference. It is both the trace scene for the whole export chain and the
reference for the gate - and it is dataset-free for consumers (HF
download; produce it yourself from a preprocessed MP3D scene with
`scripts/dump_golden.py`).

`scripts/prepare_runner_inputs.py` converts the npz + checkpoint xy grids
into the 10 runner input binaries (no dataset needed). After the runner
chain completes on those inputs, `scripts/compare_trt_golden.py` compares
the four Gaussian fields: per-field cosine >= 0.99978, then renders the
TRT Gaussians through the vendored rasterizer for a render PSNR vs the
torch render (measured: 52.95 vs 52.96 dB - lossless). The gate verdict
drives the exit code.

`scripts/verify_chain_gaussians.py` is the fast sanity variant (cosine
only, CHAIN-EQUIV-PASS/FAIL exit code).

## Smoke tier (default; structural gate)

`scripts/make_synth_inputs.py` writes seeded-random context plus
checkpoint xy grids with shapes locked to the runner contract (10 bins),
and `scripts/verify_smoke.py` checks the runner dump structurally: field
dumps exist, element counts match the runner contract, values finite.
It is a bring-up / integrity check - NOT a numerical gate. With
non-golden inputs the chain's baked golden-scene constants make the
outputs numerically meaningless.

## What the gate does not claim

- The gate is equivalence, not model accuracy - no dataset eval is
  bundled (single golden scene).
- Thresholds are the measured deployment equivalence on this code state;
  they are not statistical guarantees.
- Render PSNR is same-environment A/B only.
