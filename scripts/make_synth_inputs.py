#!/usr/bin/env python3
"""Synthetic runner inputs for dataset-free smoke runs.

Writes the 10 runner input binaries under export/runner_bins/ with the exact
shapes runner.cu expects. Context tensors are seeded random with plausible
ranges. The xy_s* grids are NOT scene data: they are gaussian-head model
state read from the checkpoint, so the smoke run exercises the same grid
state the deployed system uses.

The runner binary is locked to the 512x1024 PanSplat config and the
golden-scene candidate counts (5215/20860/83443/333772); the guards below
fail fast on mismatch (other resolution or encoder config) - rebuild
engines + runner for that config first.

Usage: python scripts/make_synth_inputs.py +experiment=pansplat-512 \
    ++model.weights_path=logs/nvpl49ge/checkpoints/last.ckpt mode=predict
"""
from pathlib import Path
import sys
import warnings

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from jaxtyping import install_import_hook

with install_import_hook(("src",), ("beartype", "beartype")):
    from hydra import initialize, compose
    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper

warnings.filterwarnings("ignore")

EXPECTED = {  # runner.cu load_bin element counts, per input
    "xy_s0": (5215, 2),
    "xy_s1": (20860, 2),
    "xy_s2": (83443, 2),
    "xy_s3": (333772, 2),
}
CTX_KEYS = ("image", "mono_image", "cube_image", "extrinsics", "near", "far")


def main() -> int:
    with initialize(config_path="../config", version_base=None):
        cfg_dict = compose(config_name="main", overrides=sys.argv[1:])
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg_dict.seed)
    torch.set_float32_matmul_precision("high")
    encoder, _ = get_encoder(cfg.model.encoder)
    model = ModelWrapper(cfg.optimizer, cfg.test, cfg.train, cfg.val,
                         cfg.predict, encoder, None, None, [],
                         None, cfg.model.weights_path, cfg.model.wo_defbp2)
    model.eval().cuda()

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
        (out / name).write_bytes(b)
        print(f"{name:16s} {str(arr.shape):20s} {len(b)} bytes")

    out = REPO / "export" / "runner_bins"
    out.mkdir(parents=True, exist_ok=True)
    write("image.bin", np.clip(rng.normal(0.5, 0.25, (1, 2, 3, 512, 1024)), 0, 1))
    write("mono_image.bin", np.clip(rng.normal(0.5, 0.25, (1, 2, 3, 512, 1024)), 0, 1))
    write("cube_image.bin", np.clip(rng.normal(0.5, 0.25, (1, 2, 3, 256, 1536)), 0, 1))
    extr = np.stack([se3(), se3()])
    write("extrinsics.bin", extr[None])
    write("near.bin", np.array([[0.4, 0.4]], dtype=np.float32))
    write("far.bin", np.array([[12.0, 12.0]], dtype=np.float32))

    # xy grids: checkpoint model state, counts locked by runner.cu buffer sizes
    shapes = {k: np.array(v.shape) for k, v in
              (("image", np.zeros((1, 2, 3, 512, 1024))),
               ("mono_image", np.zeros((1, 2, 3, 512, 1024))),
               ("cube_image", np.zeros((1, 2, 3, 256, 1536))),
               ("extrinsics", np.zeros((1, 2, 4, 4))),
               ("near", np.zeros((1, 2))), ("far", np.zeros((1, 2))))}
    for s in range(4):
        xy = getattr(model.encoder.gaussian_head, f"gs_xy_{s}_0")
        a = np.ascontiguousarray(xy.float().cpu().numpy(), dtype=np.float32
                                 ) if xy.dtype != torch.float32 else xy.cpu().numpy()
        assert tuple(a.shape) == EXPECTED[f"xy_s{s}"], (s, a.shape)
        write(f"xy_s{s}.bin", a)
        shapes[f"xy_s{s}"] = np.array(EXPECTED[f"xy_s{s}"])
    np.savez(out / "shapes.npz", **shapes)
    print("runner inputs ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
