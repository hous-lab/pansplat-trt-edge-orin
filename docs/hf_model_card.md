--- license: mit tags:
- 3d-gaussian-splatting
- panoramic
- tensorrt
- jetson
- edge-ai ---

# PanSplat TRT edge - deployment artifacts

Prebuilt artifacts for [pansplat-trt-edge-orin](https://github.com/hous-lab/pansplat-trt-edge-orin), the TensorRT deployment of PanSplat (CVPR 2025) on Jetson AGX Orin. The GitHub repo holds the pipeline; this repo holds the large regenerable artifacts so a consumer can skip ONNX tracing and go straight to engine builds and gates.

## Files

| Path | Contents | Size |
|---|---|---|
| `onnx/` | 23 graphs: the 21 deployment graphs + 2 split intermediates (manifest: `scripts/release_manifest.yaml` in the GitHub repo) | ~0.65 GB |
| `engines/x86-trt10.4/` | 21 engines, built with trtexec (TensorRT 10.4) on x86 (sm_86/sm_89 host) | ~0.3 GB |
| `engines/orin-trt10.3/` | 21 engines, built with trtexec (TensorRT 10.3) on Jetson AGX Orin (sm_87) | ~0.3 GB |
| `golden_512.npz` | golden-tier runner inputs (single MP3D scene, 512x1024), self-produced quality anchor | 331 MB |

## Usage

Clone the GitHub repo, then:

```bash
export PANSPLAT_SKIP_EXPORT=1          # use HF onnx/ instead of tracing
mkdir -p export/onnx export/engines
# copy onnx/*.onnx into export/onnx/, engines/<platform>/ into export/engines/
bash run_pipeline.sh                   # gates run against the prebuilt artifacts
```

`PANSPLAT_INPUTS=golden` plus `golden_512.npz` (placed at `export/golden_512.npz`) runs the golden tier: PSNR and cosine gates against the PyTorch reference on the anchor scene. The default synthetic tier self-anchors: the runner's inputs are generated deterministically (`scripts/make_synth_inputs.py`) and the gate compares the engine chain against the torch chain on those same inputs, so no dataset is needed anywhere.

## Platform pairing

Engines are not portable across TensorRT major versions or GPU architectures. Use `engines/orin-trt10.3/` on Jetson AGX Orin (JetPack 6.2, TensorRT 10.3), and `engines/x86-trt10.4/` on x86 hosts with TensorRT 10.4. Any other pairing: rebuild from `onnx/` with trtexec (`run_pipeline.sh` step 4 does this; the gh_s3 pair needs a large Myelin workspace, see `PANSPLAT_POOL_MB` in `docs/02_pipeline.md`).

## Provenance

- ONNX graphs: traced with the export chain in the GitHub repo from the upstream PanSplat checkpoint (`nvpl49ge`, 512x1024 mp3d tier). The checkpoint itself is upstream-hosted (link-only, see `ckpts/README.md` in the GitHub repo) and is NOT rehosted here, nor are any derivative weights.
- `golden_512.npz` is self-produced (pipeline-extracted runner inputs on the anchor scene); it contains MP3D image pixels from the single anchor scene, kept here as the quality anchor. It is not used for training.
- Engine serialization: trtexec CLI only. The TensorRT python API segfaults on these graphs; see docs/06 in the GitHub repo for the pathology notes.

## What is not here, on purpose

- Checkpoints and any dataset-derived weights: upstream-hosted, link-only.
- Runner input bins: regenerable deterministically (synthetic tier) or via the dataset prepare script (golden tier); no dataset pixels shipped except the single-scene golden anchor npz above.

Written by the deployment author; measurement contract and results: docs/03_benchmark_contract.md and docs/08_results.md in the GitHub repo. Upstream PanSplat: https://github.com/chengzhag/PanSplat (MIT, (c) Cheng Zhang).
