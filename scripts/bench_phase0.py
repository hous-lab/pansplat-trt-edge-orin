"""Phase 0 latency/VRAM baseline benchmark (reusable profiling harness).

Usage:
  cd PanSplat
  python scripts/bench_phase0.py +experiment=pansplat-512 \
      ++model.weights_path=ls933m5x/checkpoints/last.ckpt mode=predict
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

warnings.filterwarnings("ignore")


def bench(fn, warmup: int = 3, iters: int = 10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
        torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return min(times), sum(times) / len(times)


def main() -> None:
    overrides = sys.argv[1:]
    with initialize(config_path="../config", version_base=None):
        cfg_dict = compose(config_name="main", overrides=overrides)
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg_dict.seed)
    torch.set_float32_matmul_precision("high")
    import os
    if os.environ.get("PANSPLAT_CUDNN_BENCHMARK", "0") == "1":
        torch.backends.cudnn.benchmark = True
        print("== cudnn.benchmark = True ==")
    autocast = os.environ.get("PANSPLAT_AUTOCAST", "0") == "1"
    if autocast:
        print("== autocast fp16 = True ==")

    from contextlib import nullcontext

    def amp_ctx():
        return torch.autocast("cuda", dtype=torch.float16) if autocast else nullcontext()

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
        batch["context"], Tensor, lambda x: x.to("cuda"))

    # ---- encoder: forward + gaussian count + peak VRAM ----
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        gaussians = model.encoder(batch["context"], inference=True)[
            "gaussians"]["gaussians"]
    enc_peak = torch.cuda.max_memory_allocated() / 2**20
    n_gauss = gaussians.means.shape[1]

    # ---- decoder: render one pano view ----
    extrinsics = batch["context"]["extrinsics"]
    near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=1)
    far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=1)

    def render():
        with torch.inference_mode():
            return model.decoder.render_pano(
                gaussians, extrinsics[:, :1], near, far, None,
                context_extrinsics=extrinsics)

    render()  # warmup outside timing
    torch.cuda.reset_peak_memory_stats()
    def encode():
        with torch.inference_mode(), amp_ctx():
            return model.encoder(batch["context"], inference=True)

    enc_min, enc_avg = bench(encode)
    ren_min, ren_avg = bench(render)
    total_peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"gaussians        : {n_gauss}")
    print(f"encoder min/avg  : {enc_min:8.2f} / {enc_avg:8.2f} ms")
    print(f"render  min/avg  : {ren_min:8.2f} / {ren_avg:8.2f} ms")
    print(f"encoder peak VRAM: {enc_peak:8.1f} MiB")
    print(f"total   peak VRAM: {total_peak:8.1f} MiB")


if __name__ == "__main__":
    main()
