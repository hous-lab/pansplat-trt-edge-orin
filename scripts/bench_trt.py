"""TRT FP16 engine latency benchmark + numerical alignment
(TRT gaussians -> PyTorch renderer, end to end)."""
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
ENGINE = Path("export/pansplat_encoder_512_fp16.engine")
NAMES = ("image", "extrinsics", "near", "far", "mono_image", "cube_image")
OUTNAMES = ("means", "covariances", "harmonics", "opacities")


def build_model(overrides):
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
    for k in ("depth", "mask"):
        if k in batch["context"]:
            del batch["context"][k]
    ctx = apply_to_collection(batch["context"], Tensor, lambda x: x.cuda())
    return model, ctx


def main():
    import tensorrt as trt
    model, ctx = build_model(sys.argv[1:])
    attn_backend.set_backend("sdpa")
    set_sh_backend("real")

    # ---- torch golden (same backend config) ----
    with torch.inference_mode():
        g = model.encoder(ctx, inference=True)["gaussians"]["gaussians"]
    fields = ("means", "covariances", "harmonics", "opacities")
    torch_out = {k: getattr(g, k).float().cpu().numpy() for k in fields}

    # ---- TRT inference ----
    logger = trt.Logger(trt.Logger.WARNING)
    with open(ENGINE, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()
    bufs = {}
    for name in engine:
        shape = tuple(context.get_tensor_shape(name))
        dt = torch.float16 if engine.get_tensor_dtype(name) == trt.float16 else torch.float32
        bufs[name] = torch.empty(shape, dtype=dt, device="cuda")
    for n in NAMES:
        bufs[n].copy_(torch.from_numpy(torch_out[n] if False else ctx[n].cpu().numpy()))

    def infer():
        for n in NAMES + ("gaussians_means", "gaussians_covariances",
                          "gaussians_harmonics", "gaussians_opacities"):
            context.set_tensor_address(n, int(bufs[n].data_ptr()))
        context.execute_async_v3(stream.cuda_stream)

    for _ in range(5):
        infer()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(20)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(20)]
    for i in range(20):
        starts[i].record(stream)
        infer()
        ends[i].record(stream)
        torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    print(f"TRT encoder: min {min(times):.2f} / avg {sum(times)/len(times):.2f} ms")

    # ---- numerical alignment ----
    print(f"{'field':12s} {'max_abs':>10s} {'cosine':>12s}")
    trt_out = {}
    for n, k in zip(("gaussians_means", "gaussians_covariances",
                     "gaussians_harmonics", "gaussians_opacities"), fields):
        arr = bufs[n].float().cpu().numpy()
        trt_out[k] = arr
        t = torch_out[k]
        c = np.dot(t.ravel(), arr.ravel()) / (np.linalg.norm(t.ravel()) * np.linalg.norm(arr.ravel()) + 1e-12)
        print(f"{k:12s} {np.abs(t - arr).max():10.3e} {c:12.9f}")

    # ---- TRT gaussians -> PyTorch renderer end-to-end ----
    from src.model.types import Gaussians
    g_trt = Gaussians(
        means=torch.from_numpy(trt_out["means"]).cuda(),
        covariances=torch.from_numpy(trt_out["covariances"]).cuda(),
        harmonics=torch.from_numpy(trt_out["harmonics"]).cuda(),
        opacities=torch.from_numpy(trt_out["opacities"]).cuda(),
    )
    extrinsics = ctx["extrinsics"]
    near = repeat(ctx["near"][:, 0], "b -> b v", v=1)
    far = repeat(ctx["far"][:, 0], "b -> b v", v=1)
    with torch.inference_mode():
        r_trt = model.decoder.render_pano(
            g_trt, extrinsics[:, :1], near, far, None,
            context_extrinsics=extrinsics)["color"].float()
        r_t = model.decoder.render_pano(
            g, extrinsics[:, :1], near, far, None,
            context_extrinsics=extrinsics)["color"].float()
    mse = (r_t - r_trt).pow(2).mean().item()
    print(f"e2e render: mse={mse:.3e}  psnr={-10 * log10(max(mse, 1e-12)):.2f} dB")


if __name__ == "__main__":
    main()
