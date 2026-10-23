#!/usr/bin/env python3
"""Dataset-derived runner inputs: the 10 binaries runner.cu consumes.

Runs the upstream dataloader on a preprocessed pano_grf scene (the same path
dump_golden.py uses) and writes image / mono_image / cube_image / extrinsics /
near / far plus the four xy_s* depth-candidate grids to export/runner_bins/,
with a shapes.npz manifest alongside. Counterpart of make_synth_inputs.py,
which writes seeded-random inputs with the same fixed shapes for dataset-free
smoke runs.

The runner binary is locked to the 512x1024 PanSplat config and the
golden-scene candidate counts (5215/20860/83443/333772); the guards below
fail fast if the composed config yields different counts (other resolution
or encoder config) - rebuild engines + runner for that config first.
"""
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from torch import Tensor
from jaxtyping import install_import_hook

with install_import_hook(("src",), ("beartype", "beartype")):
    from hydra import initialize, compose
    from torch.utils.data import default_collate
    from lightning_fabric.utilities.apply_func import apply_to_collection
    from src.config import load_typed_root_config
    from src.dataset import get_dataset
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
CTX_KEYS = ("image", "extrinsics", "near", "far", "mono_image", "cube_image")


def main() -> None:
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
    dataset = get_dataset(cfg.dataset, "test", None)
    batch = default_collate([dataset[0]])
    ctx = apply_to_collection(batch["context"], Tensor, lambda x: x.cuda())
    out = REPO / "export" / "runner_bins"
    out.mkdir(parents=True, exist_ok=True)
    for k in CTX_KEYS:
        a = np.ascontiguousarray(ctx[k].float().cpu().numpy(), dtype=np.float32)
        a.tofile(out / f"{k}.bin")
    shapes = {}
    for k in CTX_KEYS:
        shapes[k] = np.array(ctx[k].shape)
    for s in range(4):
        xy = getattr(model.encoder.gaussian_head, f"gs_xy_{s}_0")
        a = np.ascontiguousarray(xy.float().cpu().numpy(), dtype=np.float32)
        assert tuple(a.shape) == EXPECTED[f"xy_s{s}"], (s, a.shape)
        a.tofile(out / f"xy_s{s}.bin")
        shapes[f"xy_s{s}"] = np.array(EXPECTED[f"xy_s{s}"])
    np.savez(out / "shapes.npz", **shapes)
    print("runner inputs ->", out)
    for k, v in shapes.items():
        print(" ", k, list(v))


if __name__ == "__main__":
    main()
