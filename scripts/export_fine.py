"""Export the fine-grained gaussian-head segments (gh_s2/s3 cnn+mlp) to ONNX.

Traces are taken from export/golden_512.npz (the scene the runner gates use), so
the whole export chain runs dataset-free. --only=NAME exports a single graph
(scripts/export_all.py loops --only over the 4 release graphs); OUT_DIR overrides
the output directory.
"""
import os
import sys
import warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ONLY = None
for _a in list(sys.argv):
    if _a.startswith("--only="):
        ONLY = _a.split("=", 1)[1]
        sys.argv.remove(_a)

import types
sys.modules.setdefault("wandb", types.ModuleType("wandb"))

import torch

from hydra import initialize, compose

from src.config import load_typed_root_config
from src.global_cfg import set_cfg
from src.model.decoder import get_decoder
from src.model.encoder import get_encoder
from src.model.model_wrapper import ModelWrapper
from src.model.encoder.common import attn_backend
from src.misc.sh_rotation import set_sh_backend
from src.model.encoder.gh_split import (MVSGraph, GHStageGraph,
                                        GHCnnSeg, GHMlpSeg, GHAdapterSeg)

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

g = np.load(str(ROOT / "export" / "golden_512.npz"))
ctx = {k: torch.from_numpy(g[k]).cuda()
       for k in ("image", "extrinsics", "near", "far", "mono_image", "cube_image")}

out_dir = Path(os.environ.get("OUT_DIR", "export/onnx"))
out_dir.mkdir(parents=True, exist_ok=True)

mvs = MVSGraph(model.encoder).eval().cuda()
with torch.inference_mode():
    a_out = mvs(ctx["image"], ctx["extrinsics"], ctx["near"], ctx["far"],
                ctx["mono_image"], ctx["cube_image"])
    feats, rc, dc = a_out[:4], a_out[4], a_out[5]
    pass_t = a_out[6:10]
    image, extr, near, far = pass_t

    FP16_SEG = os.environ.get("SEG_FP16", "0") == "1"   # B23: native FP16 dataflow for the mlp/ada segments

    # Run the chain, collecting real per-image trace inputs
    traces = {}
    last_raw = None

    def run_stage(s):
        global last_raw
        if s in (2, 3):
            cnn = GHCnnSeg(model.encoder, s).eval().cuda()
            mlp = GHMlpSeg(model.encoder, s).eval().cuda()
            ada = GHAdapterSeg(model.encoder, s).eval().cuda()
            if FP16_SEG:
                mlp = mlp.half()
                ada = ada.half()
            i_cnn = (feats[s], rc, image, last_raw)
            raw_cnn, images_c = cnn(*i_cnn)
            raw_cnn_s = raw_cnn.half() if FP16_SEG else raw_cnn
            images_c_s = images_c.half() if FP16_SEG else images_c
            dc_s16 = dc.half() if FP16_SEG else dc
            i_mlp = (raw_cnn_s, images_c_s, dc_s16)
            raw_g, dc_s, sh_base = mlp(*i_mlp)
            i_ada = (raw_g, dc_s, sh_base,
                     near.half() if FP16_SEG else near,
                     far.half() if FP16_SEG else far,
                     extr.half() if FP16_SEG else extr)
            outs = ada(*i_ada)
            traces[f"cnn{s}"] = i_cnn
            traces[f"mlp{s}"] = i_mlp
            traces[f"ada{s}"] = i_ada
            per_stage = outs
            last_raw = raw_cnn
        else:
            sg = GHStageGraph(model.encoder, s).eval().cuda()
            i_sg = [feats[s], rc, dc, image, extr, near, far] + \
                   ([last_raw] if s > 0 else [])
            outs = sg(*i_sg)
            traces[f"stage{s}"] = tuple(i_sg)
            per_stage = outs
            last_raw = outs[4]
        return per_stage

    for s in range(4):
        run_stage(s)
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
        print(f"EXPORT FAIL {name}: {type(e).__name__}: {str(e)[:100]}")




JOBS = {
    "mvs": lambda: export("mvs.onnx", mvs,
       (ctx["image"], ctx["extrinsics"], ctx["near"], ctx["far"],
        ctx["mono_image"], ctx["cube_image"]),
       ["image", "extrinsics", "near", "far", "mono_image", "cube_image"],
       [f"feat_{s}" for s in range(4)] +
       ["raw_correlation", "disp_candi_curr",
        "image_pass", "extrinsics_pass", "near_pass", "far_pass"]),
    "gh_stage0": lambda: export("gh_stage0.onnx", GHStageGraph(model.encoder, 0).eval().cuda(),
       traces["stage0"],
       ["features", "raw_correlation", "disp_candi_curr",
        "image", "extrinsics", "near", "far"],
       ["means_0", "covariances_0", "harmonics_0", "opacities_0", "last_raw_0"]),
    "gh_stage1": lambda: export("gh_stage1.onnx", GHStageGraph(model.encoder, 1).eval().cuda(),
       traces["stage1"],
       ["features", "raw_correlation", "disp_candi_curr",
        "image", "extrinsics", "near", "far", "last_raw"],
       ["means_1", "covariances_1", "harmonics_1", "opacities_1", "last_raw_1"]),
}
for s in (2, 3):
    JOBS[f"gh_s{s}_cnn"] = lambda s=s: export(f"gh_s{s}_cnn.onnx",
       GHCnnSeg(model.encoder, s).eval().cuda(), traces[f"cnn{s}"],
       ["features", "raw_correlation", "image", "last_raw"],
       [f"raw_cnn_{s}", f"images_c_{s}"])
    JOBS[f"gh_s{s}_mlp"] = lambda s=s: export(f"gh_s{s}_mlp.onnx",
       GHMlpSeg(model.encoder, s).eval().cuda(), traces[f"mlp{s}"],
       ["raw_cnn", "images_c", "disp_candi_curr"],
       [f"raw_g_{s}", f"dc_s_{s}", f"sh_base_{s}"])
    JOBS[f"gh_s{s}_ada"] = lambda s=s: export(f"gh_s{s}_ada.onnx",
       GHAdapterSeg(model.encoder, s).eval().cuda(), traces[f"ada{s}"],
       ["raw_g", "dc_s", "sh_base", "near", "far", "extrinsics"],
       [f"means_{s}", f"covariances_{s}", f"harmonics_{s}", f"opacities_{s}"])

if ONLY:
    JOBS[ONLY]()
else:
    for k, fn in JOBS.items():
        fn()
    print("all 9 exported to", out_dir)
