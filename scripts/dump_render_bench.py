"""Dump render bench inputs: per-face rasterizer settings + gaussian buffers."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from torch import Tensor
from jaxtyping import install_import_hook

with install_import_hook(("src",), ("beartype", "beartype")):
    from hydra import initialize, compose
    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg
    from src.model.decoder import get_decoder
import warnings
warnings.filterwarnings("ignore")

pre = sys.argv[1]
with initialize(config_path="../config", version_base=None):
    cfg_dict = compose(config_name="main", overrides=sys.argv[2:])
cfg = load_typed_root_config(cfg_dict)
set_cfg(cfg_dict)
dec = get_decoder(cfg.model.decoder, cfg.dataset).cuda().eval()

g = np.load("export/golden_512.npz")
ext = torch.from_numpy(g["extrinsics"]).cuda()
near = torch.from_numpy(g["near"]).cuda()
far = torch.from_numpy(g["far"]).cuda()
means = torch.from_numpy(np.fromfile(f"{pre}.means.f32", np.float32).reshape(1, -1, 3)).cuda()
cov = torch.from_numpy(np.fromfile(f"{pre}.cov.f32", np.float32).reshape(1, -1, 3, 3)).cuda()
harm = torch.from_numpy(np.fromfile(f"{pre}.harm.f32", np.float32).reshape(1, -1, 3, 25)).cuda()
opa = torch.from_numpy(np.fromfile(f"{pre}.opa.f32", np.float32).reshape(1, -1)).cuda()

from src.model.types import Gaussians
from src.model.decoder.cuda_splatting import GaussianRasterizer
ga = Gaussians(means=means, covariances=cov, harmonics=harm, opacities=opa)

cap = {}
import src.model.decoder.cuda_splatting as _cs
class SpyR(_cs.GaussianRasterizer):
    def __init__(self, settings):
        cap.setdefault("faces", []).append(
            {k: (v.clone() if isinstance(v, Tensor) else v)
             for k, v in settings._asdict().items()})
        super().__init__(settings)
_cs.GaussianRasterizer = SpyR

with torch.inference_mode():
    out = dec.render_pano(ga, ext, near, far, None, context_extrinsics=ext)
print("rendered; faces:", dec.num_faces, "face_w:", dec.face_w)
c = out["color"] if isinstance(out, dict) else out
print("color shape:", tuple(c.shape))

P = means.shape[1]
np.fromfile = np.fromfile  # noqa
Path("export/render_bench").mkdir(exist_ok=True)
np.frombuffer(means[0].cpu().numpy().tobytes(), dtype=np.float32).tofile("export/render_bench/means.f32")
cov[0][:, [0,0,0,1,1,2],[0,1,2,1,2,2]].cpu().numpy().tofile("export/render_bench/cov6.f32")
opa[0][..., None].cpu().numpy().tofile("export/render_bench/opa.f32")
shs_t = harm.permute(0, 1, 3, 2).contiguous()  # [1,P,3,25] -> [1,P,25,3]
shs_t[0].cpu().numpy().tofile("export/render_bench/shs.f32")
import json
n_f = len(cap["faces"])
print("captured faces:", n_f)
fbin = open("export/render_bench/faces.bin", "wb")
for f in cap["faces"]:
    for k in ("viewmatrix", "projmatrix"):
        f.cpu().numpy() if False else f[k].cpu().numpy().tofile(fbin)
    f["campos"].cpu().numpy().tofile(fbin)
    import struct
    fbin.write(struct.pack("ff", f["tanfovx"], f["tanfovy"]))
    f["bg"].cpu().numpy().tofile(fbin)
fbin.close()
meta = dict(P=int(P), face_w=int(dec.face_w), num_faces=int(dec.num_faces),
            degree=4, shs_shape=[int(x) for x in shs_t.shape[1:]])
open("export/render_bench/meta.json", "w").write(json.dumps(meta))
print("meta:", meta)
