"""Ada kernel vs captured GHAdapterSeg reference (npz), cos gate + timing."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import numpy as np
import torch
from torch.utils.cpp_extension import load

ext = load(name="ada_kernel_ext", sources=[str(REPO / "cxx" / "ada_chain_torch.cu")],
           extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)

ref = np.load(REPO / "export" / "ada_ref.npz")
torch.manual_seed(0)
ok = True
for stage in ("s2", "s3"):
    raw_g = torch.from_numpy(ref[f"raw_g_{stage}"]).cuda().half()
    dc_s = torch.from_numpy(ref[f"dc_s_{stage}"]).cuda().half()
    xy = torch.from_numpy(ref[f"xy_{stage}"]).cuda()
    near = torch.from_numpy(ref[f"near_{stage}"]).cuda().half()
    far = torch.from_numpy(ref[f"far_{stage}"]).cuda().half()
    extr = torch.from_numpy(ref[f"extr_{stage}"]).cuda().half()
    _hw = (512, 1024) if stage == "s3" else (256, 512)
    m, c, h, o = ext.ada_forward(raw_g, dc_s, xy, near, far, extr, 0.5, 15.0, *_hw)
    names = ["means", "cov", "harm", "opa"]
    outs = [m, c, h, o]
    for nm, refkey, o in zip(names, ("means","cov","harm","opa"), outs):
        r = torch.from_numpy(ref[f"{refkey}_{stage}"]).cuda().float()
        a = o.float().flatten()
        b = r.flatten()
        cs = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        md = (a - b).abs().max().item()
        print(f"{stage} {nm:6s} cos={cs:.9f} max_abs={md:.3e}")
        ok &= cs > 0.999
    hv = h.float().flatten().reshape(-1, 3, 25)
    hr = torch.from_numpy(ref[f"harm_{stage}"]).cuda().float().flatten().reshape(-1, 3, 25)
    for li, blocks in enumerate(([0], [1,2,3], [4,5,6,7,8], list(range(9,16)), list(range(16,25)))):
        a = hv[:, :, blocks].flatten()
        b = hr[:, :, blocks].flatten()
        csb = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        mdb = (a - b).abs().max().item()
        print(f"  {stage} deg{li} cos={csb:.6f} max_abs={mdb:.3e}")
print("---")
# timing
for stage in ("s2", "s3"):
    raw_g = torch.from_numpy(ref[f"raw_g_{stage}"]).cuda().half()
    for _ in range(3):
        ext.ada_forward(raw_g, dc_s, xy, near, far, extr, 0.5, 15.0, *_hw)
    ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(10):
        out = ext.ada_forward(raw_g, dc_s, xy, near, far, extr, 0.5, 15.0, *_hw)
    ev1.record(); torch.cuda.synchronize()
    print(f"{stage} kernel: {ev0.elapsed_time(ev1)/10:.2f} ms")

print("PASS" if ok else "FAIL")
