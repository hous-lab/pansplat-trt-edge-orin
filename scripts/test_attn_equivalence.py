"""B1 numerical-equivalence check: xformers (baseline) vs SDPA.

Usage:
  python scripts/test_attn_equivalence.py +experiment=pansplat-512 \
      ++model.weights_path=ls933m5x/checkpoints/last.ckpt mode=predict
"""
import sys
import warnings
from math import log10
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from einops import repeat
from jaxtyping import install_import_hook

with install_import_hook(("src",), ("beartype", "beartype")):
    from hydra import initialize, compose
    from torch import Tensor
    from torch.utils.data import default_collate
    from lightning_fabric.utilities.apply_func import apply_to_collection

    from src.config import load_typed_root_config
    from src.dataset import get_dataset
    from src.global_cfg import set_cfg
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper
    from src.model.encoder.common import attn_backend

warnings.filterwarnings("ignore")


def build(overrides):
    with initialize(config_path="../config", version_base=None):
        cfg_dict = compose(config_name="main", overrides=overrides)
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg_dict.seed)
    torch.set_float32_matmul_precision("high")
    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    model = ModelWrapper(
        cfg.optimizer, cfg.test, cfg.train, cfg.val, cfg.predict,
        encoder, None, decoder, [], None,
        cfg.model.weights_path, cfg.model.wo_defbp2,
    )
    model.eval().cuda()
    dataset = get_dataset(cfg.dataset, "test", None)
    batch = default_collate([dataset[0]])
    if "depth" in batch["context"]:
        del batch["context"]["depth"]
    if "mask" in batch["context"]:
        del batch["context"]["mask"]
    batch["context"] = apply_to_collection(
        batch["context"], Tensor, lambda x: x.cuda())
    return model, batch["context"]


def gauss_dict(g):
    return {k: getattr(g, k).detach().clone()
            for k in ("means", "covariances", "harmonics", "opacities")}


def compare(a, b):
    rows = {}
    for k in a:
        d = (a[k] - b[k]).abs()
        rel = d / b[k].abs().clamp(min=1e-6)
        cos = torch.nn.functional.cosine_similarity(
            a[k].flatten().float(), b[k].flatten().float(), dim=0)
        rows[k] = (d.max().item(), rel.median().item(), cos.item())
    return rows


def render(model, g, context):
    extrinsics = context["extrinsics"]
    near = repeat(context["near"][:, 0], "b -> b v", v=1)
    far = repeat(context["far"][:, 0], "b -> b v", v=1)
    with torch.inference_mode():
        return model.decoder.render_pano(
            g, extrinsics[:, :1], near, far, None,
            context_extrinsics=extrinsics)["color"]


def main():
    model, context = build(sys.argv[1:])
    gauss, renders = {}, {}
    for backend in ("xformers", "sdpa"):
        attn_backend.set_backend(backend)
        with torch.inference_mode():
            g = model.encoder(context, inference=True)[
                "gaussians"]["gaussians"]
        gauss[backend] = gauss_dict(g)
        renders[backend] = render(model, g, context)

    print("=== B1 numerical equivalence: xformers (baseline) vs SDPA ===")
    print(f"{'field':12s} {'max_abs':>10s} {'med_rel':>10s} {'cosine':>12s}")
    for k, (maxd, medrel, cos) in compare(gauss["xformers"], gauss["sdpa"]).items():
        print(f"{k:12s} {maxd:10.3e} {medrel:10.3e} {cos:12.9f}")
    diff = (renders["xformers"] - renders["sdpa"]).abs()
    mse = diff.pow(2).mean().item()
    psnr = -10 * log10(mse) if mse > 0 else float("inf")
    print(f"render: max_abs={diff.max().item():.3e}  mse={mse:.3e}  psnr={psnr:.2f} dB")


if __name__ == "__main__":
    main()
