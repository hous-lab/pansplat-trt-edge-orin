#!/usr/bin/env python3
"""Synthetic runner inputs for dataset-free smoke runs.

Writes the 10 runner input binaries under export/runner_bins/ with the exact
shapes runner.cu expects. Values are seeded random with plausible ranges; the
accuracy gate for these inputs is self-anchored (torch reference and TensorRT
engines see identical inputs, so the gate checks deployment equivalence, not
dataset accuracy).

The xy_s* grids keep the golden-scene candidate counts (the runner fixes
buffer sizes; a different scene needs prepare_runner_inputs.py).
"""
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "export" / "runner_bins"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20260929)


def se3():
    a = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(a)
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    T = np.eye(4)
    T[:3, :3] = q
    T[:3, 3] = rng.normal(0, 0.5, 3)
    return T


def write(name, arr):
    b = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
    (OUT / name).write_bytes(b)
    print(f"{name:16s} {str(arr.shape):20s} {len(b)} bytes")


write("image.bin", np.clip(rng.normal(0.5, 0.25, (1, 2, 3, 512, 1024)), 0, 1))
write("mono_image.bin", np.clip(rng.normal(0.5, 0.25, (1, 2, 3, 512, 1024)), 0, 1))
write("cube_image.bin", np.clip(rng.normal(0.5, 0.25, (1, 2, 3, 256, 1536)), 0, 1))

extr = np.stack([se3(), se3()])
write("extrinsics.bin", extr[None])
write("near.bin", np.array([[0.4, 0.4]], dtype=np.float32))
write("far.bin", np.array([[12.0, 12.0]], dtype=np.float32))

# xy grids: uniform pixel coords, counts fixed by runner.cu buffer sizes
for stage, n in ((0, 5215), (1, 20860), (2, 83443), (3, 333772)):
    write(f"xy_s{stage}.bin", rng.uniform(0, 1023, (n, 2)))
