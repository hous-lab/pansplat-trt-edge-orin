#!/usr/bin/env python3
"""Runner inputs from the golden anchor pack.

Writes the 10 binaries runner.cu consumes into
export/runner_bins/ plus a shapes.npz manifest.
Context tensors come from the self-produced golden
anchor pack (place the HF download at
export/golden_512.npz, no dataset needed); the xy
grids come from the checkpoint buffers
(gaussian-head model state, not scene data).
Counterpart of make_synth_inputs.py (smoke tier).

Usage: python scripts/prepare_runner_inputs.py
    +experiment=pansplat-512 mode=predict \
    ++model.weights_path=<path to last.ckpt>
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

EXPECTED = {"xy_s0": (5215, 2), "xy_s1": (20860, 2),
            "xy_s2": (83443, 2), "xy_s3": (333772, 2)}
CTX = ("image", "mono_image", "cube_image",
       "extrinsics", "near", "far")


def main() -> int:
    with initialize(config_path="../config",
                    version_base=None):
        cfg_dict = compose(config_name="main",
                           overrides=sys.argv[1:])
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg_dict.seed)
    torch.set_float32_matmul_precision("high")
    encoder, _ = get_encoder(cfg.model.encoder)
    model = ModelWrapper(
        cfg.optimizer, cfg.test, cfg.train,
        cfg.val, cfg.predict, encoder, None, None,
        [], None, cfg.model.weights_path,
        cfg.model.wo_defbp2)
    model.eval().cuda()
    gnpz = REPO / "export" / "golden_512.npz"
    if not gnpz.exists():
        print("missing", gnpz)
        print("fetch it from the Hugging Face repo,")
        print("or rerun scripts/dump_golden.py")
        return 1
    g = np.load(str(gnpz))
    out = REPO / "export" / "runner_bins"
    for k in CTX:
        a = np.ascontiguousarray(g[k], dtype=np.float32)
        a.tofile(out / (k + ".bin"))
    for s in range(4):
        gh = model.encoder.gaussian_head
        xy = getattr(gh, f"gs_xy_{s}_0")
        a = xy.cpu().numpy()
        if xy.dtype != torch.float32:
            a = a.astype(np.float32)
        a = np.ascontiguousarray(a)
        assert tuple(a.shape) == EXPECTED[f"xy_s{s}"]
        a.tofile(out / (f"xy_s{s}" + ".bin"))
    np.savez(out / "shapes.npz", **{
        "image": np.array((1, 2, 3, 512, 1024)),
        "mono_image": np.array((1, 2, 3, 512, 1024)),
        "cube_image": np.array((1, 2, 3, 256, 1536)),
        "extrinsics": np.array((1, 2, 4, 4)),
        "near": np.array((1, 2)),
        "far": np.array((1, 2)),
        "xy_s0": np.array((5215, 2)),
        "xy_s1": np.array((20860, 2)),
        "xy_s2": np.array((83443, 2)),
        "xy_s3": np.array((333772, 2))})
    print("runner inputs ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
