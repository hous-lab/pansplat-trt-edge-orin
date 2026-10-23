"""FP16 autocast vs FP32 numerical comparison (evidence for Phases 2/3).

Usage:
  python scripts/test_fp16_equivalence.py +experiment=pansplat-512 \
      ++model.weights_path=nvpl49ge/checkpoints/last.ckpt mode=predict
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

warnings.filterwarnings("ignore")


def build(overrides):
    with initialize(config_path="../config", version_base=None):
        cfg_dict = compose(config_name="main", overrides=overrides)
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg_dict.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
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
    return {k: getattr(g, k).detach().float().clone()
            for k in ("means", "covariances", "harmonics", "opacities")}


def render(model, g, context):
    extrinsics = context["extrinsics"]
    near = repeat(context["near"][:, 0], "b -> b v", v=1)
    far = repeat(context["far"][:, 0], "b -> b v", v=1)
    with torch.inference_mode():
        return model.decoder.render_pano(
            g, extrinsics[:, :1], near, far, None,
            context_extrinsics=extrinsics)["color"].float()


def main():
    model, context = build(sys.argv[1:])
    out = {}
    for name, use_amp in (("fp32", False), ("fp16_autocast", True)):
        with torch.inference_mode():
            g = model.encoder(context, inference=True) if not use_amp else None
        if use_amp:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                g = model.encoder(context, inference=True)
        out[name] = (gauss_dict(g["gaussians"]["gaussians"]),
                     render(model, g["gaussians"]["gaussians"], context))

    g32, r32 = out["fp32"]
    g16, r16 = out["fp16_autocast"]
    print("=== FP16 autocast vs FP32 (cudnn.benchmark on) ===")
    print(f"{'field':12s} {'max_abs':>10s} {'cosine':>12s}")
    for k in g32:
        d = (g32[k] - g16[k]).abs().max().item()
        c = torch.nn.functional.cosine_similarity(
            g32[k].flatten(), g16[k].flatten(), dim=0).item()
        print(f"{k:12s} {d:10.3e} {c:12.9f}")
    mse = (r32 - r16).pow(2).mean().item()
    psnr = -10 * log10(mse) if mse > 0 else float("inf")
    print(f"render: max_abs={(r32 - r16).abs().max().item():.3e}  "
          f"mse={mse:.3e}  psnr={psnr:.2f} dB")


if __name__ == "__main__":
    main()
