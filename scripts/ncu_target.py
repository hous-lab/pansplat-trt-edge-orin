"""Minimal nsys/ncu capture target: build the model and run the encoder 3x
(a target for the Nsight profilers to attach to)."""
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

overrides = sys.argv[1:]
with initialize(config_path="../config", version_base=None):
    cfg_dict = compose(config_name="main", overrides=overrides)
cfg = load_typed_root_config(cfg_dict)
set_cfg(cfg_dict)
torch.manual_seed(cfg_dict.seed)
torch.set_float32_matmul_precision("high")

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
batch["context"] = apply_to_collection(batch["context"], Tensor, lambda x: x.cuda())

with torch.inference_mode():
    for _ in range(3):
        model.encoder(batch["context"], inference=True)
torch.cuda.synchronize()
print("DONE")
