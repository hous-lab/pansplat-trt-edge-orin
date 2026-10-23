"""Export CV-split mvs graphs: mvs_pre + 3x(casc_pre, casc_post)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from einops import rearrange
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
    from src.model.encoder.costvolume.depth_predictor_cascaded import (
        warp_with_pose_depth_candidates)
import warnings
warnings.filterwarnings("ignore")
import os
FOLD = os.environ.get("FOLD", "1") == "1"
OUT = Path(os.environ.get("OUT_DIR", "export/onnx"))
OUT.mkdir(exist_ok=True)

with initialize(config_path="../config", version_base=None):
    cfg_dict = compose(config_name="main", overrides=[a for a in sys.argv[1:] if not a.startswith("--only")])
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

import os
if os.environ.get("FOLD_POSENC") == "1":
    from src.model.encoder.backbone.unimatch.position import PositionEmbeddingSine as _PES
    _orig_pe = _PES.forward
    _pe_cache = {}
    def _folded_pe(self, x, _orig=_orig_pe, _c=_pe_cache):
        key = (int(x.shape[0]), int(x.shape[2]), int(x.shape[3]))
        if key not in _c:
            _c[key] = _orig(self, torch.zeros((key[0], 1, key[1], key[2]),
                                              device=x.device, dtype=torch.float32))
        return _c[key]
    _PES.forward = _folded_pe
    print("FOLD_POSENC: position embedding folded to constants")

with torch.inference_mode():
    g = np.load("export/golden_512.npz")
    ctx = {k: torch.from_numpy(g[k]).cuda() for k in
           ("image", "extrinsics", "near", "far", "mono_image", "cube_image")}
    pre = MVSPre(encoder).eval().cuda()
    f0, f1, f2, mono = pre(ctx["image"], ctx["extrinsics"], ctx["near"],
                           ctx["far"], ctx["mono_image"], ctx["cube_image"])
    ex, nr, fr = ctx["extrinsics"], ctx["near"], ctx["far"]
    disp = None
    cp = [CascPre(encoder, s).eval().cuda() for s in range(3)]
    po = [CascPost(encoder, s).eval().cuda() for s in range(3)]
    tr = {}
    for s in range(3):
        fs = (f0, f1, f2)[s]
        args_pre = (fs, ex, nr, fr) if disp is None else (fs, ex, nr, fr, disp)
        f01, f10, dvol, pose, dc = cp[s](*args_pre)
        warped = warp_with_pose_depth_candidates(f10, pose, dvol, clamp_min_depth=1e-3,
                                                 warp_padding_mode="zeros")
        corr = (f01.unsqueeze(2) * warped).sum(1) / fs.shape[2] ** 0.5
        args_post = ((corr, f01, mono, dc) if disp is None else
                     (corr, f01, mono, dc, disp))
        disp = po[s](*args_post)
        tr[s] = dict(args_pre=args_pre, args_post=args_post)
    # A/B sanity vs full chain
    full = encoder.mvs_forward(ctx)["mvs_outputs"]
    d = (full["disp_candi_curr"] - cp[2](*tr[2]["args_pre"])[4]).abs()
    print(f"sanity dc max_abs={d.max().item():.3e}")

    def ex_(mod, args, name, ins, outs):
        torch.onnx.export(mod, args, str(OUT / name), input_names=ins,
                          output_names=outs, opset_version=17,
                          do_constant_folding=FOLD)
        print("exported", name)

    ex_(pre, (ctx["image"], ex, nr, fr, ctx["mono_image"], ctx["cube_image"]),
        "mvs_pre.onnx", ["image", "extrinsics", "near", "far", "mono_image", "cube_image"],
        ["feat_0", "feat_1", "feat_2", "mono_feat"])
    for s in range(3):
        fs = (f0, f1, f2)[s]
        pre_ins = ["features", "extrinsics", "near", "far"] + \
                  ([] if s == 0 else ["coarse_disp"])
        ex_(cp[s], tr[s]["args_pre"], f"cv_pre{s}.onnx", pre_ins,
            ["feat01", "feat10", "depth_vol", "pose", "disp_candi"])
        post_ins = ["corr", "feat01", "mono_feat", "disp_candi"] + \
                   ([] if s == 0 else ["coarse_disp"])
        ex_(po[s], tr[s]["args_post"], f"cv_post{s}.onnx", post_ins,
            ["disp_out"])
    print("ALL EXPORTED")
