# 01 — Architecture

PanSplat predicts a 3D Gaussian scene from two 512×1024 ERP images in one forward pass: 886,580 Gaussians × 88 dims (SH degree 4). The deployment task is to run that encoder as one GPU-resident chain on Orin.

## Data flow

```
inputs: image[1,2,3,512,1024] ×2 + mono/cube + extrinsics/near/far (fp16 cast at entry)
|
v
mvs_pre (backbone + mono fusion)  -> feat_0/1/2, candidate depths
|
v
cost-volume cascade (stage 0/1/2)  cv_pre0-2 -> [CV kernel] -> cv_post0-2
|
v
gaussian head, 4 stages            s{0-3}_cnn -> s{0-3}_mlp -> [adapter kernel]
|                                  (64x128 / 128x256 / 256x512 / 512x1024)
v
concat 4 stages -> 886,580 x 88    (adapter kernels write stage-2/3 gaussians
|                                 directly at offsets 52,150 / 219,036;
|                                 stage-0/1 gaussians copied D2D)
v
render: 6x256^2 cubemap rasterization -> 512x1024 ERP pano
```

## Why it is 21 engines

Each split was profile-driven. Myelin (TRT's graph compiler) failed systematically on four pathologies:

1. **36 GB fusion region** on the monolithic graph (build fails)
2. **2012-instance unrolled kernels** in the gaussian head MLPs (29.1 s → 1.69 s after replacing complex Wigner-D rotations with real-valued Rodrigues rotation, 17.7×)
3. **fp32 K=3 GEMMs forced onto FFMA + trig chains** in the adapters (818 ms → 25 ms with the adapter kernel, 32×)
4. **cost-volume graph at 70× off the bandwidth roofline** (bench: 364.3 ms → 24.5 ms with the CV kernel, 14.8×)

Each fix defines an engine boundary — **segment granularity = compiler pathology granularity**. Evolution: 1 monolith → 5 coarse graphs → 9 fine → 13 engines + 5 kernels → 21 engines + 5 kernels (flash attention splits, −29 ms). 4 legacy graphs (cv_post1/2, gh_stage0/1) remain on disk for provenance but are not loaded.

## Scene locking (feature-cache design)

This is a single-scene deployment, and the lock is explicit in the
runner: [cxx/runner.cu](../cxx/runner.cu) binds no features/raw_correlation/image inputs
to the gh_s*_cnn engines - those tensors are graph constants inside the
engines (gh_s0_cnn takes no runtime inputs at all) - and the gh_s*_mlp
engines bake the checkpoint's xy sampling grids as engine constants.
What varies at runtime: disp_candi (the cost-volume path, from the input
images), last_raw (the s1-s3 residual chain), and the xy grids on the
adapter kernels (runner input bins). Consequence for validation:
numerical gates run on the golden-anchor tier (anchor npz + checkpoint
xy, [docs/05_accuracy.md](05_accuracy.md)); the smoke tier is structural-only.


## 31-segment chain (Orin segment table, representative)

| Segments | ms | Note |
|---|---:|---|
| mvs_pre | 102.3 | copy/cast 35% of segment — data-movement bound |
| cv_pre0-2 | 3.0 | candidate depths + ERP rays |
| cvk0-2 (CV kernel) | 43.7 | 6.4 / 14.7 / 22.1 per stage |
| cv_post0-2 | 109.2 | refine + depth head; flash splits landed inside cv_post2 |
| s0 (cnn+mlp+ada) | 4.5 | |
| s1 | 17.7 | |
| s2_cnn | 46.2 | INT8 candidate segment (dropped, see [docs/08_results.md](08_results.md)) |
| s2_mlp+ada | 15.3 | |
| s3_cnn | **178.1** | largest single segment |
| s3_mlp+ada | 67.1 | s3_mlp fusion recipe on file (−5 ms, deferred) |
| concat | 0.3 | offset writes |

<p align="center">
  <img src="../assets/benchmarks/waterfall.png" width="880" alt="segment waterfall">
</p>

*Segment waterfall of the 587.8 ms-era chain (pre-flash-splits; rows sum 587.4 ms by rounding), log x-axis. Segment boundaries are those of the audited 31-segment layout (table above); the current release caliber 546.3–551.2 ms is the docs/08 record after the flash splits. Figure generated from the table above (drift-asserted maintainer-side script); no new numbers.*


- **GPU-resident, zero-file chain**: engine-to-engine tensors stay on device; no trtexec-style JSON file handoff (1693 → 1007 ms when introduced).
- Explicit buffer table (name → elements, fp32 unless `h` suffix = fp16); same-name outputs bound explicitly (`bind_out_as`) so the three disp_candi tensors don't collide.
- **fp16 in, fp32 math**: dtype convergence at boundaries via two cast kernels.
- Kernel host = graph splitting + bare kernel launches (not IPluginV3): the ada/flash kernels already sit at engine boundaries, so the runner launches them directly.
- Host-side state: `struct Eng` (engine wrapper), `load_engine`, `bind_io`, `bind_out_as`, `main` in [cxx/runner.cu](../cxx/runner.cu).
