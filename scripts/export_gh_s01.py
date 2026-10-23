"""Export gaussian-head stages 0/1 to ONNX (cnn + mlp segments; the ada
part of these stages runs as a custom CUDA kernel instead of an engine).

Usage: python scripts/export_gh_s01.py +experiment=pansplat-512 \
    ++model.weights_path=logs/nvpl49ge/checkpoints/last.ckpt mode=predict
    [--only=gh_s0_mlp]  [SEG_FP16=1: native fp16 dataflow for mlp segments]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ONLY = None
for _a in list(sys.argv):
    if _a.startswith("--only="):
        ONLY = _a.split("=", 1)[1]
        sys.argv.remove(_a)

import types
sys.modules.setdefault("wandb", types.ModuleType("wandb"))

import os
import numpy as np
import torch
from torch import Tensor, nn
from jaxtyping import install_import_hook

with install_import_hook(("src",), ("beartype", "beartype")):
    from hydra import initialize, compose
    from torch.utils.data import default_collate
    from lightning_fabric.utilities.apply_func import apply_to_collection
    from src.config import load_typed_root_config
    from src.dataset import get_dataset
    from src.global_cfg import set_cfg
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper
    from src.model.encoder.common import attn_backend
    from src.misc.sh_rotation import set_sh_backend
    from src.model.encoder.gh_split import (MVSGraph, GHStageGraph,
                                            GHCnnSeg, GHMlpSeg, GHAdapterSeg)

import warnings
warnings.filterwarnings("ignore")

with initialize(config_path="../config", version_base=None):
    cfg_dict = compose(config_name="main", overrides=sys.argv[1:])
cfg = load_typed_root_config(cfg_dict)
set_cfg(cfg_dict)
torch.manual_seed(cfg_dict.seed)
torch.set_float32_matmul_precision("high")
encoder, _ = get_encoder(cfg.model.encoder)
decoder = get_decoder(cfg.model.decoder, cfg.dataset)
model = ModelWrapper(cfg.optimizer, cfg.test, cfg.train, cfg.val, cfg.predict,
                     encoder, None, decoder, [], None,
                     cfg.model.weights_path, cfg.model.wo_defbp2)
model.eval().cuda()
attn_backend.set_backend("sdpa")
set_sh_backend("real")

gnpz = np.load(str(ROOT / "export" / "golden_512.npz"))
FP16_SEG = os.environ.get("SEG_FP16", "0") == "1"
out_dir = Path(os.environ.get("OUT_DIR", str(ROOT / "export" / "onnx")))
out_dir.mkdir(parents=True, exist_ok=True)

with torch.inference_mode():
    ctx = {k: torch.from_numpy(gnpz[k]).cuda() for k in
           ("image", "extrinsics", "near", "far", "mono_image", "cube_image")}
    mvs = MVSGraph(model.encoder).eval().cuda()
    a_out = mvs(ctx["image"], ctx["extrinsics"], ctx["near"], ctx["far"],
                ctx["mono_image"], ctx["cube_image"])
    feats, rc, dc = a_out[:4], a_out[4], a_out[5]
    image, extr, near, far = a_out[6:10]

    traces = {}
    mods = {}
    last_raw = None
    for s in (0, 1):
        cnn = GHCnnSeg(model.encoder, s).eval().cuda()
        mlp = GHMlpSeg(model.encoder, s).eval().cuda()
        ada = GHAdapterSeg(model.encoder, s).eval().cuda()
        if FP16_SEG:
            mlp = mlp.half()
        i_cnn = (feats[s], rc, image, last_raw)   # stage 0 ignores last_raw
        raw_cnn, images_c = cnn(*i_cnn)
        i_mlp = (raw_cnn.half() if FP16_SEG else raw_cnn,
                 images_c.half() if FP16_SEG else images_c,
                 dc.half() if FP16_SEG else dc)
        raw_g, dc_s, shb = mlp(*i_mlp)
        if FP16_SEG:
            i_ada = (raw_g, dc_s, shb, near.half(), far.half(), extr.half())
        else:
            i_ada = (raw_g, dc_s, shb, near, far, extr)
        ada(*i_ada)
        traces[f"cnn{s}"] = i_cnn
        traces[f"mlp{s}"] = i_mlp
        mods[s] = (cnn, mlp)
        last_raw = raw_cnn
    print("chain collected")


def export(name, module, args, in_names, out_names):
    fold = os.environ.get("FOLD", "1") == "1"
    try:
        with torch.inference_mode():
            torch.onnx.export(module, args, str(out_dir / name),
                              input_names=list(in_names),
                              output_names=list(out_names),
                              opset_version=17, do_constant_folding=fold)
        print(f"exported {name}")
    except Exception as e:
        print(f"EXPORT FAIL {name}: {type(e).__name__}: {str(e)[:120]}")
    torch.cuda.empty_cache()


JOBS = {}
for s in (0, 1):
    JOBS[f"gh_s{s}_cnn"] = lambda s=s: export(
        f"gh_s{s}_cnn.onnx", GHCnnSeg(model.encoder, s).eval().cuda(),
        traces[f"cnn{s}"],
        ["features", "raw_correlation", "image"] +
        ([] if s == 0 else ["last_raw"]),
        [f"raw_cnn_{s}", f"images_c_{s}"])
    JOBS[f"gh_s{s}_mlp"] = lambda s=s: export(
        f"gh_s{s}_mlp.onnx", mods[s][1], traces[f"mlp{s}"],
        ["raw_cnn", "images_c", "disp_candi_curr"],
        [f"raw_g_{s}", f"dc_s_{s}", f"sh_base_{s}"])

if ONLY:
    JOBS[ONLY]()
else:
    for k, fn in JOBS.items():
        fn()
    print("all 4 exported to", out_dir)