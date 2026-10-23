"""Compare trtexec --exportOutput TRT outputs against torch golden."""
import sys
import numpy as np

npz = np.load("export/golden_512.npz")
names = ["means", "covariances", "harmonics", "opacities"]
outs = {}
for i, n in enumerate(names):
    outs[n] = np.fromfile(sys.argv[i + 1], dtype=np.float32)

print(f"{'field':12s} {'max_abs':>10s} {'cosine':>12s}")
COS_GATE = 0.99978        # golden-tier contract (docs/05_accuracy.md)
PSNR_SANITY_DB = 52.0     # measured reference: 52.95 dB (docs/05_accuracy.md)
ok = True
cosines = {}
for n in names:
    t = npz["torch_" + n].ravel()
    o = outs[n]
    if o.size != t.size:
        print(f"{n}: size mismatch trt={o.size} torch={t.size}")
        ok = False
        continue
    c = float(np.dot(t, o) / (np.linalg.norm(t) * np.linalg.norm(o) + 1e-12))
    cosines[n] = c
    if c < COS_GATE:
        ok = False
    d = float(np.abs(t - o).max())
    print(f"{n:12s} {d:10.3e} {c:12.9f}")

# TRT gaussians -> local PyTorch render (the final link of the numerical-alignment chain)
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import types
sys.modules.setdefault("wandb", types.ModuleType("wandb"))
import torch
from jaxtyping import install_import_hook
with install_import_hook(("src",), ("beartype", "beartype")):
    from src.model.types import Gaussians
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
with initialize(config_path="../config", version_base=None):
    cfg_dict = compose(config_name="main", overrides=sys.argv[5:])
cfg = load_typed_root_config(cfg_dict); set_cfg(cfg_dict)
encoder, _ = get_encoder(cfg.model.encoder)
decoder = get_decoder(cfg.model.decoder, cfg.dataset)
model = ModelWrapper(cfg.optimizer, cfg.test, cfg.train, cfg.val, cfg.predict,
                     encoder, None, decoder, [], None,
                     cfg.model.weights_path, cfg.model.wo_defbp2)
model.eval().cuda()
ctx_gpu = {}
for n in ("image", "extrinsics", "near", "far", "mono_image", "cube_image"):
    ctx_gpu[n] = torch.from_numpy(npz[n]).cuda()
from einops import repeat
from src.model.types import Gaussians
g_trt = Gaussians(
    means=torch.from_numpy(
        outs["means"].reshape(npz["torch_means"].shape)).cuda(),
    covariances=torch.from_numpy(
        outs["covariances"].reshape(npz["torch_covariances"].shape)).cuda(),
    harmonics=torch.from_numpy(
        outs["harmonics"].reshape(npz["torch_harmonics"].shape)).cuda(),
    opacities=torch.from_numpy(
        outs["opacities"].reshape(npz["torch_opacities"].shape)).cuda())
extr = ctx_gpu["extrinsics"]
near = repeat(ctx_gpu["near"][:, 0], "b -> b v", v=1)
far = repeat(ctx_gpu["far"][:, 0], "b -> b v", v=1)
with torch.inference_mode():
    r_trt = model.decoder.render_pano(g_trt, extr[:, :1], near, far, None,
                                      context_extrinsics=extr)["color"].float()
    g_t = model.encoder(ctx_gpu, inference=True)["gaussians"]["gaussians"]
    r_t = model.decoder.render_pano(g_t, extr[:, :1], near, far, None,
                                    context_extrinsics=extr)["color"].float()
mse = (r_t - r_trt).pow(2).mean().item()
psnr = -10 * np.log10(max(mse, 1e-12))
print(f"e2e render(TRT gaussians vs torch): psnr = {psnr:.2f} dB")
if psnr < PSNR_SANITY_DB:
    print(f"FAIL: render PSNR {psnr:.2f} < sanity floor {PSNR_SANITY_DB} "
          "(measured reference 52.95 dB - investigate before trusting the run)")
    ok = False
verdict = "PASS" if ok else "FAIL"
print(f"GOLDEN-TIER GATE: {verdict} (cos >= {COS_GATE} per field, psnr >= {PSNR_SANITY_DB})")
sys.exit(0 if ok else 1)
