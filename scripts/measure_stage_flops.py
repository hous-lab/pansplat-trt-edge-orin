"""Per-module FLOPs attribution (input to per-module bottleneck classification)."""
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

warnings.filterwarnings("ignore")

with initialize(config_path="../config", version_base=None):
    cfg_dict = compose(config_name="main", overrides=sys.argv[1:])
cfg = load_typed_root_config(cfg_dict)
set_cfg(cfg_dict)
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

encoder, _ = get_encoder(cfg.model.encoder)
decoder = get_decoder(cfg.model.decoder, cfg.dataset)
model = ModelWrapper(cfg.optimizer, cfg.test, cfg.train, cfg.val, cfg.predict,
                     encoder, None, decoder, [], None,
                     cfg.model.weights_path, cfg.model.wo_defbp2)
model.eval().cuda()

dataset = get_dataset(cfg.dataset, "test", None)
batch = default_collate([dataset[0]])
if "depth" in batch["context"]:
    del batch["context"]["depth"]
if "mask" in batch["context"]:
    del batch["context"]["mask"]
ctx = batch["context"]
ctx = apply_to_collection(ctx, Tensor, lambda x: x.cuda())
enc = model.encoder
b, v, _, h, w = ctx["image"].shape
mvs_image = ctx["image"]
mono_erp = rearrange(ctx["mono_image"], "b v c h w -> (b v) c h w")
mono_cube = rearrange(ctx["cube_image"], "b v c h (f w) -> (b v) c h (f w)", f=2)

from torch.profiler import profile, ProfilerActivity

def flops_of(fn):
    with torch.inference_mode():
        out = fn()  # warmup + produce cached outputs
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     with_flops=True) as prof:
            fn()
        torch.cuda.synchronize()
    return sum(e.flops for e in prof.key_averages() if e.flops), out

bb = lambda: enc.backbone(mvs_image, attn_splits=enc.cfg.multiview_trans_attn_split, return_cnn_features=True)
f_bb, bb_out = flops_of(bb)

def mono_fn():
    m = enc.mono_depth(mono_erp, mono_cube)
    m["pred_depth"] = rearrange(m["pred_depth"], "(b v) 1 h w -> b v h w", b=b)
    m["mono_feat"] = rearrange(m["mono_feat"], "(b v) c h w -> b v c h w", b=b)
    return m
f_mono, md = flops_of(mono_fn)

outs = {"backbone": bb_out, "mono_depth": md}
f_cost, mv = flops_of(lambda: enc.depth_predictor(outs, ctx["extrinsics"], ctx["near"], ctx["far"]))
f_head, _ = flops_of(lambda: enc.gaussian_head(mv, ctx, None, True))

total = f_bb + f_mono + f_cost + f_head
print("== per-stage FLOPs ==")
for name, f in (("backbone", f_bb), ("mono_depth", f_mono),
                ("costvolume", f_cost), ("gaussian_head", f_head)):
    print(f"{name:16s}: {f/1e9:9.1f} GFLOPs  ({f/total*100:5.1f}%)")
print(f"{'SUM':16s}: {total/1e9:9.1f} GFLOPs")
