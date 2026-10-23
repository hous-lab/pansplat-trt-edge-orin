"""Capture GHAdapterSeg stage2/3 IO (kernel test vectors + torch reference outputs)."""
import sys, os
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from einops import rearrange
from torch import Tensor
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
(REPO / "export" / "scratch").mkdir(parents=True, exist_ok=True)
encoder, _ = get_encoder(cfg.model.encoder)
decoder = get_decoder(cfg.model.decoder, cfg.dataset)
model = ModelWrapper(cfg.optimizer, cfg.test, cfg.train, cfg.val, cfg.predict,
                     encoder, None, decoder, [], None,
                     cfg.model.weights_path, cfg.model.wo_defbp2)
model.eval().cuda()
attn_backend.set_backend("sdpa")
set_sh_backend("real")

with torch.inference_mode():
    mvs = MVSGraph(model.encoder).eval().cuda()
with torch.inference_mode():
    import numpy as np
    gnpz = np.load(str(Path(__file__).resolve().parent.parent / "export" / "golden_512.npz"))
    ctx = {k: torch.from_numpy(gnpz[k]).cuda() for k in
           ("image", "extrinsics", "near", "far", "mono_image", "cube_image")}

    mvs = MVSGraph(model.encoder).eval().cuda()
    a_out = mvs(ctx["image"], ctx["extrinsics"], ctx["near"], ctx["far"],
                ctx["mono_image"], ctx["cube_image"])
    feats, rc, dc = a_out[:4], a_out[4], a_out[5]
    image, extr, near, far = a_out[6:10]
    last_raw = None
    for s in (0, 1):
        sg = GHStageGraph(model.encoder, s).eval().cuda()
        i_sg = [feats[s], rc, dc, image, extr, near, far] + \
               ([last_raw] if s > 0 else [])
        outs = sg(*i_sg)
        last_raw = outs[4]
        if s == 1:
            last_raw.float().cpu().numpy().tofile(str(REPO / "export" / "scratch" / "torch_lastraw1.f32"))
            np.save(REPO / "export" / "scratch" / "ada_ref_lastraw1.npy", last_raw.float().cpu().numpy())
    rec = {}
    for s in (2, 3):
        cnn = GHCnnSeg(model.encoder, s).eval().cuda()
        mlp = GHMlpSeg(model.encoder, s).eval().cuda()
        ada = GHAdapterSeg(model.encoder, s).eval().cuda()
        mlp = mlp.half()
        ada = ada.half()
        i_cnn = (feats[s], rc, image, last_raw)
        raw_cnn, images_c = cnn(*i_cnn)
        raw_cnn_s = raw_cnn.half()
        images_c_s = images_c.half()
        dc_s16 = dc.half()
        i_mlp = (raw_cnn_s, images_c_s, dc_s16)
        raw_g, dc_s, sh_base = mlp(*i_mlp)
        i_ada = (raw_g, dc_s, sh_base, near.half(), far.half(), extr.half())
        outs = ada(*i_ada)
        rec[f"s{s}"] = dict(
            raw_g=raw_g.float().cpu().numpy(),
            dc_s=dc_s.float().cpu().numpy(),
            sh_base=np.zeros(0, np.float32),
            near=near.float().cpu().numpy(),
            far=far.float().cpu().numpy(),
            extr=extr.float().cpu().numpy(),
            means=outs[0].float().cpu().numpy(),
            cov=outs[1].float().cpu().numpy(),
            harm=outs[2].float().cpu().numpy(),
            opa=outs[3].float().cpu().numpy(),
            xy=ada.xy.float().cpu().numpy(),
            raw_cnn=raw_cnn.float().cpu().numpy(),
        )
        last_raw = raw_cnn

    np.savez(REPO / "export" / "ada_ref.npz", **{f"{k}_{sname}": v for sname, d in rec.items() for k, v in d.items()})
    print("saved export/ada_ref.npz")
    for sname, d in rec.items():
        print(sname, {k: v.shape for k, v in d.items()})

