#!/bin/bash
# PanSplat TRT edge deployment pipeline - 8 idempotent steps.
#
# Usage:
#   bash run_pipeline.sh                          # full run (synthetic inputs)
#   PANSPLAT_INPUTS=golden bash run_pipeline.sh   # golden MP3D scene tier
#   PANSPLAT_SKIP_EXPORT=1 bash run_pipeline.sh   # reuse prebuilt ONNX (HF)
#
# Steps write content sentinels under export/.ok/ so re-runs skip completed
# work; delete export/.ok/<step> to force rework of that step.
set -euo pipefail
cd "$(dirname "$0")"

log() { printf "\n=== [%s] %s ===\n" "$(date +%H:%M:%S)" "$1"; }
die() { echo "FATAL: $1" >&2; exit 1; }

# ---- step 1: environment guard ------------------------------------------
log "step 1/8 guard"
python3 scripts/gen_release_manifest.py --check
CKPT=logs/nvpl49ge/checkpoints/last.ckpt
# Torch-side producers (dump_golden / inputs / export) share one hydra config;
# mode=predict keeps the unimatch weights loader out of the import path.
OVERRIDES="+experiment=pansplat-512 mode=predict ++model.weights_path=$CKPT"
[ -f "$CKPT" ] || die "missing $CKPT - download per ckpts/README.md"
command -v trtexec >/dev/null || die "trtexec not on PATH - engines are built
with the TensorRT CLI only (the pip python API segfaults on these graphs);
source the TRT env or add its bin/ to PATH"
python3 -c 'import torch; assert torch.__version__.startswith("2.4.0"), \
  "torch 2.4.0+cu118 required (environment_strict.yaml)"'

OKDIR=export/.ok
mkdir -p "$OKDIR" export/scratch

# ---- step 2: runner inputs -----------------------------------------------
log "step 2/8 runner inputs"
if [ ! -f "$OKDIR/02_inputs" ]; then
  # golden_512.npz anchors the golden tier: fetch it from the Hugging Face
  # repo, or produce it from a preprocessed MP3D scene (dump_golden.py).
  # The smoke tier skips the npz but still needs the checkpoint (xy grids
  # are gaussian-head model state, not scene data).
  if [ "${PANSPLAT_INPUTS:-synth}" = "golden" ] && [ ! -f export/golden_512.npz ]; then
    echo "producing export/golden_512.npz (torch trace scene)"
    python3 scripts/dump_golden.py $OVERRIDES
  fi
  if [ "${PANSPLAT_INPUTS:-synth}" = "golden" ]; then
    python3 scripts/prepare_runner_inputs.py $OVERRIDES
  else
    python3 scripts/make_synth_inputs.py $OVERRIDES
  fi
  touch "$OKDIR/02_inputs"
fi

# ---- step 3: ONNX export --------------------------------------------------
log "step 3/8 ONNX export"
if [ ! -f "$OKDIR/03_onnx" ]; then
  if [ "${PANSPLAT_SKIP_EXPORT:-0}" = "1" ]; then
    echo "SKIP_EXPORT: validating HF-fetched ONNX against the manifest"
    python3 scripts/engine_build_list.py > /dev/null
  else
    python3 scripts/export_all.py $OVERRIDES
  fi
  touch "$OKDIR/03_onnx"
fi

# ---- step 4: TensorRT engines ----------------------------------------------
log "step 4/8 TensorRT engines"
if [ ! -f "$OKDIR/04_engines" ]; then
  PLAT_DIR=$(python3 scripts/detect_platform.py)
  ENG_DIR=export/engines/$PLAT_DIR
  mkdir -p "$ENG_DIR"
  echo "platform: $PLAT_DIR"
  POOL_MB="${PANSPLAT_POOL_MB:-12288}"
  RAM_GB=$(free -g | awk '/^Mem:/{print $7}')
  NEED_GB=$((POOL_MB / 1024 + 4))
  for e in gh_s3_cnn gh_s3_mlp; do
    if [ ! -f "$ENG_DIR/$e.engine" ] && [ ! -f "$ENG_DIR/$e.engine.ok" ]; then
      if [ "$RAM_GB" -lt "$NEED_GB" ]; then
        die "building $e needs ~${NEED_GB}GB free RAM (Myelin workspace pool ${POOL_MB}MB, override with PANSPLAT_POOL_MB), found ${RAM_GB}GB free - build on a big-RAM host and copy into export/engines/"
      fi
    fi
  done
  while IFS="|" read -r onnx engine flags; do
    [ -n "$onnx" ] || continue
    if [ ! -f "$ENG_DIR/$engine" ] && [ ! -f "$ENG_DIR/$engine.ok" ]; then
      echo "building $engine"
      trtexec --onnx="export/onnx/$onnx" --saveEngine="$ENG_DIR/$engine" \
        --fp16 $flags > "export/scratch/trtexec_$engine.log" 2>&1 || \
        die "trtexec failed for $engine - see export/scratch/trtexec_$engine.log"
      touch "$ENG_DIR/$engine.ok"
    fi
  done < <(python3 scripts/engine_build_list.py)
  trtexec --version > "$OKDIR/04_engines" 2>/dev/null
  echo "engines in $ENG_DIR"
fi

# ---- step 5: build cxx binaries --------------------------------------------
log "step 5/8 build cxx"
if [ ! -f "$OKDIR/05_cxx" ]; then
  make -C cxx all
  touch "$OKDIR/05_cxx"
fi

# ---- step 6: accuracy gates --------------------------------------------------
log "step 6/8 accuracy gates"
if [ ! -f "$OKDIR/06_gates" ]; then
  PLAT_DIR=$(python3 scripts/detect_platform.py)
  if [ "${PANSPLAT_INPUTS:-synth}" = "golden" ]; then
    ./runner export/runner_bins export/engines/$PLAT_DIR export/scratch/golden_run
    python3 scripts/compare_trt_golden.py export/scratch/golden_run.means.f32 \
      export/scratch/golden_run.cov.f32 \
      export/scratch/golden_run.harm.f32 \
      export/scratch/golden_run.opa.f32 $OVERRIDES
  else
    echo "smoke tier: structural gate only - the chain bakes the golden"
    echo "scene encoding (see docs/05_accuracy.md); numerical validation ="
    echo "golden tier (PANSPLAT_INPUTS=golden)"
    ./runner export/runner_bins export/engines/$PLAT_DIR export/scratch/synth_run
    python3 scripts/verify_smoke.py export/scratch/synth_run
  fi
  touch "$OKDIR/06_gates"
fi


# ---- step 7: latency bench -------------------------------------------------
log "step 7/8 latency bench"
if [ ! -f "$OKDIR/07_bench" ]; then
  PLAT_DIR=$(python3 scripts/detect_platform.py)
  if [ ! -x "./runner" ]; then
    die "runner binary missing (step 5 must produce ./runner)"
  fi
  if [ "$(uname -m)" = "aarch64" ]; then
    echo "Orin discipline: MAXN + quiet window, min-of-N"
    echo "(measurement contract: docs/03_benchmark_contract.md)"
    nvpmodel -q 2>/dev/null | head -1 || true
    ./runner export/runner_bins export/engines/$PLAT_DIR export/scratch/bench_run
    echo "repeat for min-of-N statistics; see docs/03_benchmark_contract.md"
  else
    ./runner export/runner_bins export/engines/$PLAT_DIR export/scratch/bench_run
    echo "x86: chain timing above; per-engine numbers via scripts/bench_trt.py"
  fi
  touch "$OKDIR/07_bench"
fi

# ---- step 8: summary banner -------------------------------------------------
log "step 8/8 summary"
echo "pipeline complete. Expected reference numbers (see docs/03_benchmark_contract.md):"
echo "  Orin AGX 64GB (JetPack 6.2, TRT 10.3, MAXN): reconstruction 546.3-551.2 ms"
echo "  render (6x256^2 cubemap -> ERP):             20.6 ms"
echo "  end-to-end:                                  ~567 ms"
echo "  accuracy (golden tier): cos >= 0.99978, render PSNR 52.95 dB"
echo "  smoke tier (default): structural gate only - the chain bakes"
echo "    the golden scene; accuracy gate = golden tier: cos >= 0.99978,"
echo "    render PSNR ~52.95 dB (PANSPLAT_INPUTS=golden)"
echo "artifacts: export/onnx (21), export/engines/$PLAT_DIR (21), cxx binaries"
touch "$OKDIR/08_banner"
echo "ALL DONE"
