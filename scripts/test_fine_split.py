"""Equivalence check: fine-grained three-segment split (cnn/mlp/adapter) vs the
original forward (stages 2/3)."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import types
sys.modules.setdefault("wandb", types.ModuleType("wandb"))

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

dataset = get_dataset(cfg.dataset, "test", None)
batch = default_collate([dataset[0]])
for k in ("depth", "mask"):
    if k in batch["context"]:
        del batch["context"][k]
ctx = apply_to_collection(batch["context"], Tensor, lambda x: x.cuda())

with torch.inference_mode():
    g = model.encoder(ctx, inference=True)["gaussians"]["gaussians"]
ref = {k: getattr(g, k).detach().clone() for k in
       ("means", "covariances", "harmonics", "opacities")}

mvs = MVSGraph(model.encoder).eval().cuda()
with torch.inference_mode():
    a_out = mvs(ctx["image"], ctx["extrinsics"], ctx["near"], ctx["far"],
                ctx["mono_image"], ctx["cube_image"])
    feats, rc, dc = a_out[:4], a_out[4], a_out[5]
    image, extr, near, far = a_out[6], a_out[7], a_out[8], a_out[9]

    # last_raw for stages 2/3 must be re-derived: the code above set it to None for
    # 2/3, which was wrong
    # Re-run with last_raw passed through correctly
    per_stage, last_raw = [], None
    for s in range(4):
        if s in (2, 3):
            cnn = GHCnnSeg(model.encoder, s).eval().cuda()
            mlp = GHMlpSeg(model.encoder, s).eval().cuda()
            ada = GHAdapterSeg(model.encoder, s).eval().cuda()
            raw_cnn, images_c = cnn(feats[s], rc, image, last_raw)
            raw_g, dc_s, sh_base = mlp(raw_cnn, images_c, dc)
            outs = ada(raw_g, dc_s, sh_base, near, far, extr)
            per_stage.append(outs)
            last_raw = raw_cnn     # cross-stage state = CNN segment output
        else:
            sg = GHStageGraph(model.encoder, s).eval().cuda()
            outs = sg(feats[s], rc, dc, image, extr, near, far, last_raw)
            per_stage.append(outs[:4])
            last_raw = outs[4]

print("=== fine-grained three-segment (2/3) vs original ===")
ok = True
for i, k in enumerate(("means", "covariances", "harmonics", "opacities")):
    a = torch.cat([p[i] for p in per_stage], dim=1)
    r = ref[k]
    d = (a - r).abs()
    cos = torch.nn.functional.cosine_similarity(
        a.flatten().float(), r.flatten().float(), dim=0).item()
    print(f"{k:12s} max_abs={d.max().item():.3e}  cos={cos:.9f}")
    ok &= cos > 0.99999
print("PASS" if ok else "FAIL")
