"""Segmented export wrappers: encoder = graph A (mvs) + a single-stage gaussian-head
chain + three fine-grained segments inside each stage.

v2: the fine-grained segment classes (GHCnnSeg / GHMlpSeg / GHAdapterSeg) hold
deep-copied minimal submodule references,
so .half() works without polluting the shared encoder (fixes the recursive-Half
pitfall hit during B23).
"""
import copy

import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn

from src.model.encoder.gaussian_head import unpad_pano


class MVSGraph(nn.Module):
    """Graph A: backbone + mono + cost volume (= mvs_forward)."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, image, extrinsics, near, far, mono_image, cube_image):
        context = {"image": image, "extrinsics": extrinsics, "near": near,
                   "far": far, "mono_image": mono_image, "cube_image": cube_image}
        out = self.encoder.mvs_forward(context)["mvs_outputs"]
        outs = [s.get("features") for s in out["stages"]]
        outs.append(out["raw_correlation"])
        outs.append(out["disp_candi_curr"])
        outs.extend([image, extrinsics, near, far])
        return tuple(outs)


class GHStageGraph(nn.Module):
    """Graph B-s: single-stage GaussianHead (whole segment; used for stages 0/1)."""

    def __init__(self, encoder, stage_idx):
        super().__init__()
        self.encoder = encoder
        self.gh = encoder.gaussian_head
        self.stage_idx = stage_idx

    def forward(self, features, raw_correlation, disp_candi_curr,
                image, extrinsics, near, far, last_raw=None):
        gh = self.gh
        b, v, c, H, W = image.shape
        images_fullres = rearrange(image, "b v c h w -> (v b) c h w")
        rc = rearrange(raw_correlation, "b v ... -> (v b) ...")
        dc = rearrange(disp_candi_curr, "b v ... -> (v b) ...")
        stage_idx = self.stage_idx

        if features is not None:
            feats = rearrange(features, "b v ... -> (v b) ...")
            features_c = gh.crop_patch(feats, stage_idx, 0, "features")
        else:
            features_c = None
        images_c = gh.crop_patch(images_fullres, stage_idx, 0, "images")
        rc_c = gh.crop_patch(rc, stage_idx, 0, "raw_correlation")

        xy = getattr(gh, f"gs_xy_{stage_idx}_0")
        full_grid = repeat(xy, "n xy -> vb n 1 xy", vb=int(v) * int(b))

        ins = [images_c, rc_c]
        if features_c is not None:
            ins.insert(1, features_c)
        raw_in = torch.cat(ins, dim=1)

        if stage_idx > 0 and not gh.wo_pgs_res:
            lr = torch.nn.functional.interpolate(last_raw, scale_factor=2,
                                                 mode="bilinear")
            lr = unpad_pano(lr, gh.padding)
            raw_in = torch.cat([raw_in, lr], dim=1)

        delta = gh.to_gaussians_list[stage_idx](raw_in)
        if stage_idx == 0 or gh.wo_pgs_res:
            raw = delta
        else:
            raw = lr + delta
        new_last_raw = raw

        sh_res = ((stage_idx == 0 and not gh.wo_sh_res)
                  or (not gh.wo_sh_res and gh.wo_pgs_res))
        dc_s = torch.nn.functional.grid_sample(dc, full_grid, padding_mode="border")
        dc_s = rearrange(dc_s, "(v b) d n 1 -> b v n d", v=v, b=b)
        patch_grid = repeat(gh.map_patch_xy(xy, stage_idx, 0),
                            "n xy -> vb n 1 xy", vb=int(v) * int(b))
        raw_g = torch.nn.functional.grid_sample(raw, patch_grid, padding_mode="border")
        raw_g = rearrange(raw_g, "(v b) c n 1 -> b v n c", v=v, b=b)
        raw_g = gh.gaussians_mlp_list[stage_idx](raw_g)
        if sh_res:
            sh_base = torch.nn.functional.grid_sample(images_c, patch_grid,
                                                      padding_mode="border")
            sh_base = rearrange(sh_base, "(v b) c n 1 -> b v n c", v=v, b=b)
        else:
            sh_base = None

        rg, rc_out, rho = raw_g.split(
            [gh.gaussian_raw_channels, rc.shape[1], 1], dim=-1)
        pdf = torch.softmax(rc_out, dim=-1)
        fine_disp = (dc_s * pdf).sum(dim=-1, keepdim=True)
        densities = torch.sigmoid(rho)
        fine_disp = fine_disp.clamp(
            1.0 / rearrange(far, "b v -> b v () ()"),
            1.0 / rearrange(near, "b v -> b v () ()"))
        depths = 1.0 / fine_disp

        g = gh.convert_gaussians(depths, densities, rg, extrinsics,
                                 xy, gh.full_shape[stage_idx], None,
                                 sh_base if sh_res else None)
        return (g.means, g.covariances, g.harmonics, g.opacities, new_last_raw)


class GHCnnSeg(nn.Module):
    """Segment 1: crop + CNN (+residual) -> raw_cnn and images_c (shared gh; verified exportable)."""

    def __init__(self, encoder, stage_idx):
        super().__init__()
        self.encoder = encoder
        self.gh = encoder.gaussian_head
        self.stage_idx = stage_idx

    def forward(self, features, raw_correlation, image, last_raw):
        gh = self.gh
        b, v, c, H, W = image.shape
        stage_idx = self.stage_idx
        if features is not None:
            feats = rearrange(features, "b v ... -> (v b) ...")
            features_c = gh.crop_patch(feats, stage_idx, 0, "features")
        else:
            features_c = None
        images_c = gh.crop_patch(images_fullres := rearrange(
            image, "b v c h w -> (v b) c h w"), stage_idx, 0, "images")
        rc_c = gh.crop_patch(rearrange(raw_correlation, "b v ... -> (v b) ..."),
                             stage_idx, 0, "raw_correlation")
        xy = getattr(gh, f"gs_xy_{stage_idx}_0")
        ins = [images_c, rc_c]
        if features_c is not None:
            ins.insert(1, features_c)
        raw_in = torch.cat(ins, dim=1)
        if stage_idx > 0 and not gh.wo_pgs_res:
            lr = torch.nn.functional.interpolate(last_raw, scale_factor=2,
                                                 mode="bilinear")
            lr = unpad_pano(lr, gh.padding)
            raw_in = torch.cat([raw_in, lr], dim=1)
        raw_cnn = gh.to_gaussians_list[stage_idx](raw_in)
        if not (stage_idx == 0 or gh.wo_pgs_res):
            raw_cnn = lr + raw_cnn
        return raw_cnn, images_c


class GHMlpSeg(nn.Module):
    """Segment 2: fibo grid_sample + MLP -> raw_g / dc_s / sh_base. Minimal references."""

    def __init__(self, encoder, stage_idx):
        super().__init__()
        gh = encoder.gaussian_head
        self.mlp = copy.deepcopy(gh.gaussians_mlp_list[stage_idx])
        self.register_buffer("xy", getattr(gh, f"gs_xy_{stage_idx}_0").clone())
        self.register_buffer("range_xy",
                             getattr(gh, f"range_xy_{stage_idx}_0").clone())
        self.stage_idx = stage_idx

    def forward(self, raw_cnn, images_c, disp_candi_curr):
        raw_cnn = raw_cnn.to(self.xy.dtype)
        images_c = images_c.to(self.xy.dtype)
        disp_candi_curr = disp_candi_curr.to(self.xy.dtype)
        b, v = disp_candi_curr.shape[:2]
        dc = rearrange(disp_candi_curr, "b v ... -> (v b) ...")
        xy = self.xy
        full_grid = repeat(xy, "n xy -> vb n 1 xy", vb=raw_cnn.shape[0])
        patch_size = self.range_xy[:, 1] - self.range_xy[:, 0]
        patch_grid = (xy - self.range_xy[:, 0]) / patch_size * 2 - 1
        patch_grid = repeat(patch_grid, "n xy -> vb n 1 xy", vb=raw_cnn.shape[0])
        dc_s = torch.nn.functional.grid_sample(dc, full_grid, padding_mode="border")
        dc_s = rearrange(dc_s, "(v b) d n 1 -> b v n d", v=v, b=b)
        raw_g = torch.nn.functional.grid_sample(raw_cnn, patch_grid,
                                                padding_mode="border")
        raw_g = rearrange(raw_g, "(v b) c n 1 -> b v n c", v=v, b=b)
        raw_g = self.mlp(raw_g)
        sh_res = self.stage_idx == 0
        if sh_res:
            sh_base = torch.nn.functional.grid_sample(images_c, patch_grid,
                                                      padding_mode="border")
            sh_base = rearrange(sh_base, "(v b) c n 1 -> b v n c", v=v, b=b)
        else:
            sh_base = torch.zeros(1, device=raw_g.device, dtype=raw_g.dtype)
        return raw_g, dc_s, sh_base


class GHAdapterSeg(nn.Module):
    """Segment 3: split/softmax/depths + convert_gaussians (inlined) -> 4 gaussian params.
Minimal references."""

    def __init__(self, encoder, stage_idx):
        super().__init__()
        gh = encoder.gaussian_head
        self.adapter = copy.deepcopy(gh.gaussian_adapter)
        self.register_buffer("xy", getattr(gh, f"gs_xy_{stage_idx}_0").clone())
        self.full_shape = tuple(gh.full_shape[stage_idx])
        self.num_surfaces = gh.num_surfaces
        self.wo_fibo_gs = gh.wo_fibo_gs
        self.stage_idx = stage_idx

    def forward(self, raw_g, dc_s, sh_base, near, far, extrinsics):
        xy = self.xy
        h, w = self.full_shape
        srf = self.num_surfaces
        device = raw_g.device

        rg, rc_out, rho = raw_g.split(
            [raw_g.shape[-1] - dc_s.shape[-1] - 1, dc_s.shape[-1], 1], dim=-1)
        pdf = torch.softmax(rc_out, dim=-1)
        fine_disp = (dc_s * pdf).sum(dim=-1, keepdim=True)
        densities = torch.sigmoid(rho)
        fine_disp = fine_disp.clamp(
            1.0 / rearrange(far, "b v -> b v () ()"),
            1.0 / rearrange(near, "b v -> b v () ()"))
        depths = 1.0 / fine_disp

        densities = repeat(densities, "b v n dpt -> b v n srf dpt", srf=srf)
        depths = repeat(depths, "b v n dpt -> b v n srf dpt", srf=srf)

        xy_ray = rearrange(xy, "n xy -> n 1 xy") / 2 + 0.5
        gaussians = rearrange(rg, "... (srf c) -> ... srf c", srf=srf)
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), device=device).type_as(xy)
        if not self.wo_fibo_gs:
            lat = xy[..., 1] * np.pi / 2
            r = torch.clamp(torch.cos(lat), min=1e-2)
            pixel_width = 1 / r
            pixel_size = torch.stack((pixel_width, torch.ones_like(pixel_width)),
                                     dim=-1) * pixel_size
            pixel_size = rearrange(pixel_size, "n xy -> n 1 xy")
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size

        g = self.adapter.forward(
            rearrange(extrinsics, "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v n 1 xy -> b v n 1 () xy"),
            depths,
            densities,
            rearrange(gaussians[..., 2:], "b v n srf c -> b v n srf () c"),
            (h, w),
            sh_base=rearrange(sh_base, "b v n c -> b v n () () c")
            if sh_base is not None and sh_base.dim() >= 3 else None,
        )
        keys = ("means", "covariances", "harmonics", "opacities")
        outs = []
        for k in keys:
            t = getattr(g, k)
            if t.dim() >= 4:
                t = rearrange(t, "b v n srf spp ... -> b (v n srf spp) ...")
            outs.append(t)
        return tuple(outs)
