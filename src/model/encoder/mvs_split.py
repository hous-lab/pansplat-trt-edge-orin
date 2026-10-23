"""CV split: MVSPre (backbone + mono) + CascPre (depth candidates + rays) /
CascPost (refine + depth head).

Statement-for-statement the same order as depth_predictor_cascaded.forward; the CV
sites are externalized as runner kernels.
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn

from src.model.encoder.costvolume.depth_predictor_cascaded import (
    prepare_feat_proj_data_lists,
)


class MVSPre(nn.Module):
    """Backbone transformer + mono fusion -> feat_0/1/2 + mono_feat."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, image, extrinsics, near, far, mono_image, cube_image):
        b, v, _, h, w = image.shape
        mvs_image = image
        if h > self.encoder.cfg.fpn_max_height:
            mvs_image = rearrange(image, "b v c h w -> (b v) c h w")
            mvs_image = F.interpolate(
                mvs_image, size=(self.encoder.cfg.fpn_max_height,
                                 self.encoder.cfg.fpn_max_height * 2),
                mode="bilinear")
            mvs_image = rearrange(mvs_image, "(b v) c h w -> b v c h w", b=b)
        backbone_out = self.encoder.backbone(
            mvs_image, attn_splits=self.encoder.cfg.multiview_trans_attn_split,
            return_cnn_features=True)
        mono_erp = rearrange(mono_image, "b v c h w -> (b v) c h w")
        mono_cube = rearrange(cube_image, "b v c h (f w) -> (b v) c h (f w)", f=2)
        md = self.encoder.mono_depth(mono_erp, mono_cube)
        mono_feat = rearrange(md["mono_feat"], "(b v) c h w -> b v c h w", b=b)
        feats = list(backbone_out["trans_features"])
        return feats[0], feats[1], feats[2], mono_feat


class CascPre(nn.Module):
    """Depth candidates + ERP ray prep -> CV kernel inputs
(feat01, feat10, depth_vol, pose, dc)."""

    def __init__(self, encoder, stage_idx):
        super().__init__()
        self.stage_idx = stage_idx

    def forward(self, features, extrinsics, near, far, coarse_disp=None):
        b, v, c, h, w = features.shape
        h, w = int(h), int(w)
        ndc = [128, 64, 32][self.stage_idx]
        feat_comb_lists, pose_curr_lists = prepare_feat_proj_data_lists(
            features, extrinsics)
        if self.stage_idx == 0:
            min_depth = rearrange(1.0 / far.clone().detach(), "b v -> (v b) 1")
            max_depth = rearrange(1.0 / near.clone().detach(), "b v -> (v b) 1")
            disp_candi = (
                min_depth + torch.linspace(0.0, 1.0, ndc).unsqueeze(0).to(min_depth.device)
                * (max_depth - min_depth)
            ).type_as(features)
            disp_candi = repeat(disp_candi, "vb d -> vb d h w", h=h, w=w)
            disp_interval = (max_depth - min_depth) / (ndc - 1)
        else:
            min_depth = rearrange(1.0 / far.clone().detach(), "b v -> (v b) 1")
            max_depth = rearrange(1.0 / near.clone().detach(), "b v -> (v b) 1")
            disp_interval = (max_depth - min_depth) / ([128, 64, 32][0] - 1) / (2 ** self.stage_idx)
            shift = torch.linspace(-ndc // 2, ndc // 2 - 1, ndc).type_as(features)
            shift = repeat(shift, "d -> vb d h w", vb=v * b, h=h, w=w)
            disp_curr = F.interpolate(coarse_disp, scale_factor=2, mode='bilinear',
                                      align_corners=True).detach()
            disp_candi = (
                disp_curr + disp_interval.view(disp_curr.shape[0], 1, 1, 1) * shift
            ).clamp(min=min_depth.unsqueeze(-1).unsqueeze(-1),
                    max=max_depth.unsqueeze(-1).unsqueeze(-1))
        feat01 = feat_comb_lists[0]
        feat10 = feat_comb_lists[1]
        pose = pose_curr_lists[0]
        return feat01, feat10, (1.0 / disp_candi).contiguous(), pose, disp_candi


class CascPost(nn.Module):
    """corr -> concat(feat01, disp, mono) -> refine+residual+head -> softmax -> disparity."""

    def __init__(self, encoder, stage_idx):
        super().__init__()
        dp = encoder.depth_predictor
        self.stage_idx = stage_idx
        self.refine = copy.deepcopy(dp.corr_refine_nets[stage_idx])
        self.residual = copy.deepcopy(dp.regressor_residuals[stage_idx])
        self.head = copy.deepcopy(dp.depth_heads[stage_idx])

    def forward(self, corr, feat01, mono_feat, dc, coarse_disp=None, debug_raw=False):
        raw_in = torch.cat((corr, feat01), dim=1)
        if self.stage_idx >= 1:
            disp_curr = F.interpolate(coarse_disp, scale_factor=2, mode='bilinear',
                                      align_corners=True).detach()
            raw_in = torch.cat((raw_in, disp_curr), dim=1)
        mono = rearrange(mono_feat, "b v c h w -> (v b) c h w")
        mono = F.interpolate(mono, size=raw_in.shape[-2:], mode='bilinear',
                             align_corners=False)
        raw_in = torch.cat([raw_in, mono], dim=1)
        raw = self.refine(raw_in)
        raw = raw + self.residual(raw_in)
        raw = self.head(raw)
        pdf = F.softmax(raw, dim=1)
        if debug_raw:
            return (dc * pdf).sum(dim=1, keepdim=True), raw
        return (dc * pdf).sum(dim=1, keepdim=True)
