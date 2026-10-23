"""Dump torch golden: inputs + encoder gaussian outputs (for TRT numerical comparison)."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import types
sys.modules.setdefault("wandb", types.ModuleType("wandb"))  # B11

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
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper
    from src.model.encoder.common import attn_backend
    from src.misc.sh_rotation import set_sh_backend

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

out = {}
for n in ("image", "extrinsics", "near", "far", "mono_image", "cube_image"):
    out[n] = ctx[n].float().cpu().numpy()
for k in ("means", "covariances", "harmonics", "opacities"):
    out["torch_" + k] = getattr(g, k).float().cpu().numpy()
np.savez("export/golden_512.npz", **out)
print("saved export/golden_512.npz:",
      {k: v.shape for k, v in out.items() if k.startswith("torch_")})
