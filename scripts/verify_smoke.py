#!/usr/bin/env python3
"""Smoke-tier gate: structural checks on the runner dump (no torch).

The 21-engine chain bakes the golden-scene encoding into the gh_s*_cnn
engines (runner.cu binds no features/correlation/image to them;
gh_s0_cnn takes no inputs at all) and the xy grids into the mlp engines.
With non-golden inputs the outputs are not numerically meaningful, so the
smoke tier checks only structure: every stage field dump exists with the
declared element count and finite values. Numerical validation runs on
the golden-anchor tier (PANSPLAT_INPUTS=golden): compare_trt_golden.py
compares the runner dump against the torch anchors in the golden npz.
"""
import sys
from pathlib import Path
import numpy as np

PRE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("export/scratch/synth_run")
COUNTS = [5215, 20860, 83443, 333772]
DIMS = {"means": 3, "cov": 6, "harm": 45, "opa": 1}
ok = True
for fld, dim in DIMS.items():
    f = Path(str(PRE) + "." + fld + ".f32")
    if not f.exists():
        print("FAIL missing", f)
        ok = False
        continue
    v = np.fromfile(f, dtype=np.float32)
    want = sum(COUNTS) * dim
    if v.size != want:
        print("FAIL", f.name, v.size, "!=", want, "floats")
        ok = False
        continue
    fin = bool(np.isfinite(v).all())
    if not fin:
        print("FAIL", f.name, "non-finite values")
        ok = False
    print(f"{f.name:18s} {v.size:9d} floats, finite: {fin}")
print("SMOKE-TIER GATE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
