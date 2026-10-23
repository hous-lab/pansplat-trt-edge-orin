"""Render runner gaussians vs golden torch gaussians, image PSNR gate.

Usage: render_runner_gaussians.py <gaussian_prefix> [weights=nvpl49ge]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from torch import Tensor
from jaxtyping import install_import_hook

with install_import_hook(("src",), ("beartype", "beartype")):
    from hydra import initialize, compose
    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg
    from src.model.decoder import get_decoder
    from src.model.model_wrapper import ModelWrapper
import warnings
warnings.filterwarnings("ignore")

pre = sys.argv[1]
with initialize(config_path="../config", version_base=None):
    cfg_dict = compose(config_name="main", overrides=sys.argv[2:])
cfg = load_typed_root_config(cfg_dict)
set_cfg(cfg_dict)
dec = get_decoder(cfg.model.decoder, cfg.dataset).cuda().eval()

g = np.load("export/golden_512.npz")
ext = torch.from_numpy(g["extrinsics"]).cuda()
near = torch.from_numpy(g["near"]).cuda()
far = torch.from_numpy(g["far"]).cuda()

def to_gauss(prefix, src):
    if src == "golden":
        return (torch.from_numpy(g["torch_means"]).cuda(),
                torch.from_numpy(g["torch_covariances"]).cuda(),
                torch.from_numpy(g["torch_harmonics"]).cuda(),
                torch.from_numpy(g["torch_opacities"]).cuda())
    means = torch.from_numpy(np.fromfile(f"{pre}.means.f32", np.float32).reshape(1, -1, 3)).cuda()
    cov = torch.from_numpy(np.fromfile(f"{pre}.cov.f32", np.float32).reshape(1, -1, 3, 3)).cuda()
    harm = torch.from_numpy(np.fromfile(f"{pre}.harm.f32", np.float32).reshape(1, -1, 3, 25)).cuda()
    opa = torch.from_numpy(np.fromfile(f"{pre}.opa.f32", np.float32)).reshape(1, -1).cuda()
    return means, cov, harm, opa

def render(params):
    m, c, h, o = params

    from src.model.types import Gaussians
    gg = Gaussians(means=m, covariances=c, harmonics=h, opacities=o)
    with torch.inference_mode():
        img = dec.render_pano(gg, ext[:, :1], near[:, :1], far[:, :1], None,
                              context_extrinsics=ext)
    return (img["color"] if isinstance(img, dict) else img).clamp(0, 1)

im_gold = render(to_gauss(pre, "golden"))
im_run = render(to_gauss(pre, "runner"))
mse = ((im_gold - im_run) ** 2).mean().item()
psnr = -10 * np.log10(mse + 1e-12)
print(f"render PSNR(runner vs golden) = {psnr:.2f} dB")
import torchvision.utils as vutils
import os
os.makedirs("outputs/phase4", exist_ok=True)
vutils.save_image(im_run[0], "outputs/phase4/render_runner.png")
vutils.save_image(im_gold[0], "outputs/phase4/render_golden.png")
print("PASS" if psnr > 30 else "FAIL", "(gate: >30dB; reference value 49.14dB, see docs)")
