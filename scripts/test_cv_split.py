"""Torch A/B: CV-split chain vs full mvs_forward (dc bit-exact gate)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from einops import rearrange
import torch.nn.functional as F
from jaxtyping import install_import_hook

with install_import_hook(("src",), ("beartype", "beartype")):
    from hydra import initialize, compose
    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg
    from src.model.model_wrapper import ModelWrapper
    from src.model.encoder import get_encoder
    from src.model.decoder import get_decoder
    from src.model.encoder.common import attn_backend
    from src.misc.sh_rotation import set_sh_backend
    from src.model.encoder.mvs_split import MVSPre, CascPre, CascPost
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

with torch.inference_mode():
    g = np.load("export/golden_512.npz")
    ctx = {k: torch.from_numpy(g[k]).cuda() for k in
           ("image", "extrinsics", "near", "far", "mono_image", "cube_image")}
    # full path
    full = encoder.mvs_forward(ctx)["mvs_outputs"]
    dc_full = full["disp_candi_curr"]
    # split path (torch CV for tracing parity)
    from src.model.encoder.costvolume.depth_predictor_cascaded import (
        warp_with_pose_depth_candidates)
    pre = MVSPre(encoder).eval().cuda()
    f0, f1, f2, mono = pre(ctx["image"], ctx["extrinsics"], ctx["near"],
                           ctx["far"], ctx["mono_image"], ctx["cube_image"])
    disp = None
    corr_trace = []
    USE_FULL_DISP0 = True

    for s in range(3):
        cp = CascPre(encoder, s).eval().cuda()
        po = CascPost(encoder, s).eval().cuda()
        args = (f0, f1, f2, mono) if False else (f0, f1, f2, mono)
        feats = (f0, f1, f2)
        disp_in = disp
        if s == 1 and USE_FULL_DISP0:
            disp_in = rearrange(full["stages"][0]["disp"], "b v h w -> (v b) 1 h w")
        f01, f10, dvol, pose, dc = cp(feats[s], ctx["extrinsics"], ctx["near"],
                                      ctx["far"], disp_in)
        warped = warp_with_pose_depth_candidates(f10, pose, dvol,
                                                 clamp_min_depth=1e-3,
                                                 warp_padding_mode="zeros")
        corr = (f01.unsqueeze(2) * warped).sum(1) / feats[s].shape[2] ** 0.5
        corr_trace.append((f01, f10, dvol, pose, dc, corr))
        disp = po(corr, f01, mono, dc, disp)
    dc_split = corr_trace[2][4]
    dc_split = corr_trace[2][4]
    for s in range(3):
        dc_s_full = full["stages"][s]["disp_candi_curr"]
        dc_s_split = corr_trace[s][4]
        d = (dc_s_full - dc_s_split).abs()
        print(f"stage{s} dc: max_abs={d.max().item():.3e}")
    d = (dc_full - dc_split).abs()
    print(f"final dc: max_abs={d.max().item():.3e} "
          f"cos={torch.nn.functional.cosine_similarity(dc_full.flatten(), dc_split.flatten(), dim=0).item():.9f}")
    print("PASS" if d.max().item() == 0 else "CHECK", "(gate: max_abs=0)")
