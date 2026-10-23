"""Model statistics: per-module parameter counts + single-forward encoder FLOPs
(input to the Orin roofline estimate)."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
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

encoder, _ = get_encoder(cfg.model.encoder)
decoder = get_decoder(cfg.model.decoder, cfg.dataset)
model = ModelWrapper(cfg.optimizer, cfg.test, cfg.train, cfg.val, cfg.predict,
                     encoder, None, decoder, [], None,
                     cfg.model.weights_path, cfg.model.wo_defbp2)
model.eval().cuda()

def n_params(m):
    return sum(p.numel() for p in m.parameters())

total = n_params(model)
print("== parameter counts ==")
print(f"model total : {total/1e6:.2f} M")
for name in ("backbone", "mono_depth", "depth_predictor", "gaussian_head"):
    if hasattr(model.encoder, name):
        m = getattr(model.encoder, name)
        print(f"encoder.{name:16s}: {n_params(m)/1e6:8.2f} M")
print(f"decoder      : {n_params(model.decoder)/1e6:8.3f} M")

dataset = get_dataset(cfg.dataset, "test", None)
batch = default_collate([dataset[0]])
if "depth" in batch["context"]:
    del batch["context"]["depth"]
if "mask" in batch["context"]:
    del batch["context"]["mask"]
batch["context"] = apply_to_collection(batch["context"], Tensor, lambda x: x.cuda())

with torch.inference_mode():
    model.encoder(batch["context"], inference=True)  # warmup
torch.cuda.synchronize()

from torch.profiler import profile, ProfilerActivity
with torch.inference_mode():
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 with_flops=True) as prof:
        model.encoder(batch["context"], inference=True)
    torch.cuda.synchronize()

total_flops = 0
rows = []
for ev in prof.key_averages():
    if ev.flops and ev.flops > 0:
        rows.append((ev.flops, ev.key))
        total_flops += ev.flops
rows.sort(reverse=True)
print("\n== FLOPs (single encoder forward) ==")
print(f"total: {total_flops/1e9:.1f} GFLOPs")
for f, k in rows[:6]:
    print(f"  {f/1e9:9.2f} GFLOPs  {k[:60]}")
