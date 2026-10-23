"""ONNX (ORT) vs PyTorch numerical-alignment check.

Usage:
  python scripts/test_onnx_equivalence.py +experiment=pansplat-512 \
      ++model.weights_path=nvpl49ge/checkpoints/last.ckpt mode=predict
"""
import sys
import warnings
from math import log10
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import types
sys.modules.setdefault("wandb", types.ModuleType("wandb"))  # B11: predict does not use wandb

import numpy as np
import torch
from einops import repeat
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
ONNX_PATH = "export/pansplat_encoder_512.onnx"


def build_model(overrides):
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
    for k in ("depth", "mask"):
        if k in batch["context"]:
            del batch["context"][k]
    ctx = apply_to_collection(batch["context"], Tensor, lambda x: x.cuda())
    return model, ctx


def render(model, g, context):
    extrinsics = context["extrinsics"]
    near = repeat(context["near"][:, 0], "b -> b v", v=1)
    far = repeat(context["far"][:, 0], "b -> b v", v=1)
    with torch.inference_mode():
        return model.decoder.render_pano(
            g, extrinsics[:, :1], near, far, None,
            context_extrinsics=extrinsics)["color"].float()


def main():
    import onnxruntime as ort
    model, ctx = build_model(sys.argv[1:])
    attn_backend.set_backend("sdpa")
    set_sh_backend("real")

    with torch.inference_mode():
        g = model.encoder(ctx, inference=True)["gaussians"]["gaussians"]
    fields = ("means", "covariances", "harmonics", "opacities")
    torch_out = {k: getattr(g, k).float().cpu().numpy() for k in fields}
    inputs = {n: ctx[n].float().cpu().numpy()
              for n in ("image", "extrinsics", "near", "far",
                        "mono_image", "cube_image")}

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    iname = [i.name for i in sess.get_inputs()]
    oname = [o.name for o in sess.get_outputs()]
    ort_out = sess.run(oname, {k: inputs[k] for k in iname})

    print("=== ONNX(ORT-CPU) vs PyTorch(sdpa+real-SH) ===")
    print(f"{'field':12s} {'max_abs':>10s} {'cosine':>12s}")
    for k, o in zip(fields, ort_out):
        t = torch_out[k]
        d = np.abs(t - o).max()
        c = np.dot(t.ravel(), o.ravel()) / (
            np.linalg.norm(t.ravel()) * np.linalg.norm(o.ravel()) + 1e-12)
        print(f"{k:12s} {d:10.3e} {c:12.9f}")

    # Render comparison: ORT gaussians -> PyTorch renderer
    import onnx
    m = onnx.load(ONNX_PATH)
    onnx.checker.check_model(m)

    def to_g(o):
        from src.model.types import Gaussians
        return Gaussians(
            means=torch.from_numpy(o[0]).float().cuda(),
            covariances=torch.from_numpy(o[1]).float().cuda(),
            harmonics=torch.from_numpy(o[2]).float().cuda(),
            opacities=torch.from_numpy(o[3]).float().cuda(),
        )

    r_ort = render(model, to_g(ort_out), ctx)
    r_t = render(model, g, ctx)
    mse = (r_t - r_ort).pow(2).mean().item()
    print(f"render: mse={mse:.3e}  psnr={-10 * log10(max(mse, 1e-12)):.2f} dB")


if __name__ == "__main__":
    main()
