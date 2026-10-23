"""Local side: render transferred TRT gaussian outputs with the PyTorch renderer,
and compare against golden."""
import sys
import warnings
from math import log10
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import types
sys.modules.setdefault("wandb", types.ModuleType("wandb"))  # B11

import numpy as np
import torch
from torch import Tensor
from einops import repeat
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
    from src.model.types import Gaussians

warnings.filterwarnings("ignore")

with initialize(config_path="../config", version_base=None):
    cfg_dict = compose(config_name="main", overrides=sys.argv[1:])
cfg = load_typed_root_config(cfg_dict)
set_cfg(cfg_dict)
torch.manual_seed(cfg_dict.seed)
encoder, _ = get_encoder(cfg.model.encoder)
decoder = get_decoder(cfg.model.decoder, cfg.dataset)
model = ModelWrapper(cfg.optimizer, cfg.test, cfg.train, cfg.val, cfg.predict,
                     encoder, None, decoder, [], None,
                     cfg.model.weights_path, cfg.model.wo_defbp2)
model.eval().cuda()

dataset = get_dataset(cfg.dataset, "test", None)
batch = default_collate([dataset[0]])
for k in ("depth", "mask"):
    if k in batch["context"]:
        del batch["context"][k]
ctx = apply_to_collection(batch["context"], Tensor, lambda x: x.cuda())

SCRATCH = Path(__file__).resolve().parent.parent / "export" / "scratch"
trt_out = np.load(SCRATCH / "trt_out.npz")
d_golden = np.load("export/golden_512.npz")
def shaped(k):
    return torch.from_numpy(trt_out[k].reshape(d_golden["torch_" + k].shape)).float().cuda()
g_trt = Gaussians(
    means=shaped("means"),
    covariances=shaped("covariances"),
    harmonics=shaped("harmonics"),
    opacities=shaped("opacities"))

extr = ctx["extrinsics"]
near = repeat(ctx["near"][:, 0], "b -> b v", v=1)
far = repeat(ctx["far"][:, 0], "b -> b v", v=1)
with torch.inference_mode():
    g = model.encoder(ctx, inference=True)["gaussians"]["gaussians"]
    r_trt = model.decoder.render_pano(g_trt, extr[:, :1], near, far, None,
                                      context_extrinsics=extr)["color"].float()
    r_t = model.decoder.render_pano(g, extr[:, :1], near, far, None,
                                    context_extrinsics=extr)["color"].float()
mse = (r_t - r_trt).pow(2).mean().item()
print(f"end-to-end: TRT gaussians -> PyTorch render vs PyTorch full chain: PSNR = {-10 * log10(max(mse, 1e-12)):.2f} dB")

from PIL import Image
img = (r_trt[0, 0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
Image.fromarray(img).save(SCRATCH / "trt_render.png")
print("saved export/scratch/trt_render.png")
