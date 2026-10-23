# 07 — Deployment guide

## Platform pairing

`scripts/detect_platform.py` names the engine directory by architecture + TensorRT version (e.g. `x86-trt10.4`). Engines are not portable across pairings and the name prevents silent reuse: build per platform, keep per-platform directories separate. Pairings measured for this contract: Orin AGX (sm_87, TRT 10.3), A6000 (sm_86, TRT 10.16), 4090 Laptop dev host (sm_89/x86, TRT 10.4).

## Orin

```bash
# JetPack 6.2, TRT 10.3 on PATH (trtexec)
nvpmodel -m 0 && jetson_clocks   # MAXN
PANSPLAT_INPUTS=golden bash run_pipeline.sh   # accuracy gate; smoke is default
```

Orin's 64 GB host RAM builds all 21 engines locally, including the gh_s3 pair. Follow docs/03 discipline for any number you publish.

## x86 with dGPU

- Python env: `environment_strict.yaml` (python 3.10.16, torch 2.4.0+cu118, TRT 10.x with trtexec on PATH)
- The gh_s3 pair needs ~41 GB free RAM during trtexec (Myelin workspace). Step 4 hard-guards this; on small-RAM hosts it tells you to build on a big-RAM host and copy engines into `export/engines/<platform>/`.
- x86 engines are x86-only — Orin builds its own from the same ONNX.

## HF users (no source build)

```bash
PANSPLAT_SKIP_EXPORT=1 bash run_pipeline.sh
```

Validates the 21 HF-fetched ONNX against the manifest, builds engines with the local trtexec, builds the runner, runs the smoke gate (structural only). For the numerical
gate fetch the anchor npz and run the golden tier - still dataset-free.

## Runner invocation

```bash
./runner <bin_dir> <engines_dir> <out_prefix>
```

e.g. `./runner export/runner_bins export/engines/x86-trt10.4 export/scratch/g_run` writes `<prefix>.{means,cov,harm,opa}.f32` gaussian dumps. Gate scripts take the dump prefix as their argument. Cold start (engine load + context init) is 3.1–3.6 s, separate from steady-state latency.
