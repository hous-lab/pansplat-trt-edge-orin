#!/usr/bin/env python3
"""Drive the full ONNX export chain: all 21 deployment graphs.

The producers run in order as isolated subprocesses (fail-fast):
  1. export_cv_split.py   -> mvs_pre, cv_pre0/1/2, cv_post0 (+ cv_post1/2 intermediates)
  2. subprocess export_split_post.py on cv_post1.onnx -> p1_{a,m1,m2,b}
  3. export_split_post.py on cv_post2.onnx -> p2_{a,m1,m2,b}
  4. export_gh_s01.py     -> gh_s{0,1}_{cnn,mlp}
  5. export_fine.py       -> gh_s{2,3}_{cnn,mlp} (one --only invocation per graph)

Hydra overrides are forwarded to every torch-side producer, e.g.
  python3 scripts/export_all.py +experiment=pansplat-512 \
      ++model.weights_path=logs/nvpl49ge/checkpoints/last.ckpt mode=predict
All torch-side producers trace with export/golden_512.npz as the input scene,
so no dataset is needed at export time.
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "export" / "onnx"
LOGDIR = REPO / "export" / "scratch" / "export_all"
LOGDIR.mkdir(parents=True, exist_ok=True)
ARGS = sys.argv[1:]

STEPS = [
    ("cv_split", [sys.executable, str(REPO / "scripts" / "export_cv_split.py")], True),
    # graph surgery: no torch/hydra - hydra overrides must NOT be forwarded
    ("p1_split", [sys.executable, str(REPO / "scripts" / "export_split_post.py"),
                  str(OUT / "cv_post1.onnx"), str(OUT), "p1"], False),
    ("p2_split", [sys.executable, str(REPO / "scripts" / "export_split_post.py"),
                  str(OUT / "cv_post2.onnx"), str(OUT), "p2"], False),
    ("gh_s01", [sys.executable, str(REPO / "scripts" / "export_gh_s01.py")], True),
]
for g in ("gh_s2_cnn", "gh_s2_mlp", "gh_s3_cnn", "gh_s3_mlp"):
    STEPS.append((g, [sys.executable, str(REPO / "scripts" / "export_fine.py"),
                      f"--only={g}"], True))

def main():
    for name, cmd, needs_args in STEPS:
        log = LOGDIR / f"{name}.log"
        print(f"[export_all] {name} -> {log}")
        with open(log, "w") as fh:
            r = subprocess.run(cmd + (ARGS if needs_args else []), cwd=REPO,
                               stdout=fh, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(open(log).read()[-4000:], file=sys.stderr)
            sys.exit(f"export step failed: {name} (log: {log})")
        print(f"[export_all] {name} done")
    n = len(list(OUT.glob("*.onnx")))
    print(f"[export_all] {n} ONNX under {OUT} (expect 23: 21 final graphs + "
          f"cv_post1/cv_post2 split inputs)")

if __name__ == "__main__":
    main()
