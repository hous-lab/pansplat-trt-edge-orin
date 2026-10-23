"""Numerical-equivalence check: CV kernel vs a PyTorch reference (4090)."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

warnings.filterwarnings("ignore")

ext = load(name="cv_kernel_ext", sources=["cxx/cv_kernel_torch.cu"],
           extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)

torch.manual_seed(0)
vb, C, D, H, W = 2, 128, 128, 64, 128
feat_ref = torch.randn(vb, C, H, W).cuda().half()
feat_src = torch.randn(vb, C, H, W).cuda().half()
depth_t = (torch.rand(vb, D, H, W).cuda() * 2 + 0.45)  # depth (the kernel interface takes depth directly)
pose = torch.eye(4)[None].expand(vb, 4, 4).clone().cuda()
pose[:, :3, :3] = torch.linalg.qr(torch.randn(vb, 3, 3))[0].cuda().float()
pose[:, :3, 3] = torch.randn(vb, 3).cuda() * 0.1

# PyTorch reference (equation-by-equation replica of warp_with_pose_depth_candidates + correlation)
def reference(feat_ref, feat_src, disp, pose):
    import math
    b = feat_ref.shape[0]
    u = torch.linspace(0.5 / W, 1 - 0.5 / W, W, device=feat_ref.device)
    v = torch.linspace(0.5 / H, 1 - 0.5 / H, H, device=feat_ref.device)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    phi = uu * 2 * np.pi - np.pi
    theta = vv * np.pi - np.pi / 2
    pts = torch.stack([torch.cos(theta) * torch.sin(phi),
                       torch.sin(theta),
                       torch.cos(theta) * torch.cos(phi)])  # [3,H,W]
    pts = pts[None].repeat(b, 1, 1, 1).reshape(b, 3, -1)  # [b,3,HW]
    outs = []
    for bi in range(b):
        R = pose[bi, :3, :3]
        t = pose[bi, :3, 3]
        p = (R @ pts) * 1.0
        p = p.unsqueeze(2).repeat(1, 1, D, 1) * depth_t[bi].reshape(1, 1, D, -1)
        p = p + t[None, :, None, None]
        nrm = p.norm(p=2, dim=1, keepdim=True).clamp(min=1e-3)
        p = p / nrm
        phi2 = torch.atan2(p[:, 0], p[:, 2])
        theta2 = torch.asin(p[:, 1].clamp(-1, 1))
        u2 = (phi2 + np.pi) / (2 * np.pi)
        v2 = (theta2 + np.pi / 2) / np.pi
        grid = torch.stack([u2 * 2 - 1, v2 * 2 - 1], dim=-1)  # [b, D*H, W, 2]
        grid = grid.reshape(b, D, H, W, 2).reshape(b, D * H, W, 2)
        warped = F.grid_sample(feat_src[bi:bi+1].float(), grid[bi:bi+1],
                               mode="bilinear", padding_mode="zeros",
                               align_corners=True)
        warped = warped.reshape(C, D, H, W).unsqueeze(0)
        corr = (feat_ref[bi:bi+1].unsqueeze(2) * warped).sum(1) / math.sqrt(C)
        outs.append(corr)
    return torch.cat(outs)

with torch.inference_mode():
    # Ground truth: the repo-original warp + correlation
    sys.path.insert(0, '.')
    from src.model.encoder.costvolume.depth_predictor_cascaded import warp_with_pose_depth_candidates
    from einops import rearrange
    fc = feat_src.float()
    corr_parts = []
    for bi in range(feat_ref.shape[0]):
        w = warp_with_pose_depth_candidates(
            fc[bi:bi+1], pose[bi:bi+1], depth_t[bi:bi+1],  # the model passes 1.0/disp = depth; the kernel keeps the same semantics
            clamp_min_depth=1e-3, warp_padding_mode="zeros")
        fr = feat_ref[bi:bi+1].float()
        c = (fr.unsqueeze(2) * w).sum(1) / np.sqrt(feat_ref.shape[1])
        corr_parts.append(c)
    ref = torch.cat(corr_parts)
    ker = ext.cost_volume(feat_ref, feat_src, depth_t, pose).float()

d = (ref - ker).abs()
cos = torch.nn.functional.cosine_similarity(ref.flatten(), ker.flatten(), dim=0).item()
print(f"CV kernel vs PyTorch reference: max_abs={d.max().item():.3e}  "
      f"mean_abs={d.mean().item():.3e}  cos={cos:.9f}")
print("PASS" if cos > 0.999 and d.max().item() < 0.5 else "FAIL")
