"""Per-stage encoder latency breakdown + torch.profiler kernel hotspots (Phase 2 entry).

Usage:
  python scripts/bench_stages.py +experiment=pansplat-512 \
      ++model.weights_path=ls933m5x/checkpoints/last.ckpt mode=predict
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from einops import rearrange
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


def bench(fn, warmup=3, iters=10):
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


def main():
    model, context = build(sys.argv[1:])
    enc = model.encoder
    b, v, _, h, w = context["image"].shape

    # Reproduce the mvs_forward input prep (at 512 the height equals fpn_max_height,
    # no interpolation needed)
    mvs_image = context["image"]
    mono_erp = rearrange(context["mono_image"], "b v c h w -> (b v) c h w")
    mono_cube = rearrange(
        context["cube_image"], "b v c h (f w) -> (b v) c h (f w)", f=2)
    extrinsics = context["extrinsics"]
    near, far = context["near"], context["far"]

    for backend in ("xformers", "sdpa"):
        attn_backend.set_backend(backend)
        # Precompute the per-stage input caches once
        with torch.inference_mode():
            bb = enc.backbone(
                mvs_image, attn_splits=enc.cfg.multiview_trans_attn_split,
                return_cnn_features=True)
            md = enc.mono_depth(mono_erp, mono_cube)
            md["pred_depth"] = rearrange(md["pred_depth"], "(b v) 1 h w -> b v h w", b=b)
            md["mono_feat"] = rearrange(md["mono_feat"], "(b v) c h w -> b v c h w", b=b)
            outs = {"backbone": bb, "mono_depth": md}
            mv = enc.depth_predictor(outs, extrinsics, near, far)

            def f_backbone():
                return enc.backbone(
                    mvs_image, attn_splits=enc.cfg.multiview_trans_attn_split,
                    return_cnn_features=True)

            def f_mono():
                m = enc.mono_depth(mono_erp, mono_cube)
                m["pred_depth"] = rearrange(m["pred_depth"], "(b v) 1 h w -> b v h w", b=b)
                m["mono_feat"] = rearrange(m["mono_feat"], "(b v) c h w -> b v c h w", b=b)
                return m

            def f_cost():
                return enc.depth_predictor(outs, extrinsics, near, far)

            def f_head():
                return enc.gaussian_head(mv, context, None, True)

            t_bb = bench(f_backbone)
            t_mono = bench(f_mono)
            t_cost = bench(f_cost)
            t_head = bench(f_head)
            t_all = bench(lambda: enc(context, inference=True))
        total = sum(x[1] for x in (t_bb, t_mono, t_cost, t_head))
        print(f"\n=== stage breakdown [{backend}] ===")
        print(f"{'stage':16s} {'min ms':>9s} {'avg ms':>9s} {'share':>7s}")
        for name, t in (("backbone", t_bb), ("mono_depth", t_mono),
                        ("costvolume", t_cost), ("gaussian_head", t_head)):
            print(f"{name:16s} {t[0]:9.2f} {t[1]:9.2f} {t[1]/total*100:6.1f}%")
        print(f"{'SUM(stages)':16s} {'':9s} {total:9.2f}")
        print(f"{'encoder(e2e)':16s} {'':9s} {t_all[1]:9.2f}")
    return model, context


def profile_kernels(model, context, backend, topk=12):
    from torch.profiler import profile, ProfilerActivity
    attn_backend.set_backend(backend)
    for _ in range(2):
        with torch.inference_mode():
            model.encoder(context, inference=True)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            with torch.inference_mode():
                model.encoder(context, inference=True)
        torch.cuda.synchronize()
    print(f"\n=== top-{topk} CUDA kernels [{backend}] ===")
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=topk))


if __name__ == "__main__":
    if "--profile" in sys.argv[1:]:
        args = [a for a in sys.argv[1:] if a != "--profile"]
        model, context = build(args)
        torch.cuda.empty_cache()
        profile_kernels(model, context, "xformers")
        torch.cuda.empty_cache()
        profile_kernels(model, context, "sdpa")
    else:
        main()
