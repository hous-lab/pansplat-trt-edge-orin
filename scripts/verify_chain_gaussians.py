"""Quick compare of a runner gaussian dump against export/golden_512.npz.

The full golden gate (4-field cos + render PSNR) lives in
scripts/compare_trt_golden.py; this is the small fast sanity check.
Usage: verify_chain_gaussians.py [dump-prefix]
"""
from pathlib import Path
import sys
import numpy as np

REPO = Path(__file__).resolve().parent.parent
pre = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "export" / "scratch" / "g_run")
g = np.load(REPO / "export" / "golden_512.npz")
keys = {"means": "torch_means", "cov": "torch_covariances",
        "harm": "torch_harmonics", "opa": "torch_opacities"}
print(f"{'field':6s} {'max_abs':>10s} {'cosine':>12s}")
ok = True
for k, gk in keys.items():
    a = np.fromfile(f"{pre}.{k}.f32", dtype=np.float32)
    r = g[gk].astype(np.float32).reshape(-1)
    if a.size != r.size:
        print(k, "SIZE MISMATCH", a.size, r.size); ok = False; continue
    a = a.astype(np.float64); r = r.astype(np.float64)
    cs = float(np.dot(a, r) / (np.linalg.norm(a) * np.linalg.norm(r) + 1e-12))
    md = float(np.abs(a - r).max())
    print(f"{k:6s} {md:10.3e} {cs:12.9f}")
    ok &= cs > 0.99978
print("CHAIN-EQUIV-PASS" if ok else "CHAIN-EQUIV-FAIL")
sys.exit(0 if ok else 1)
