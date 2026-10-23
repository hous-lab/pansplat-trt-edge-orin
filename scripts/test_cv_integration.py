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

import types
from torch.utils.cpp_extension import load
ext = load(name="cv_kernel_ext", sources=["cxx/cv_kernel_torch.cu"],
           extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)


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


def cv_loop_cuda(self, inputs, extrinsics, near, far):
    """DepthPredictorCascaded.forward with fused CUDA cost-volume correlation."""
    from einops import rearrange, repeat
    import torch.nn.functional as F
    from src.model.encoder.costvolume.depth_predictor_cascaded import (
        prepare_feat_proj_data_lists,
    )
    outputs = {}
    outputs["stages"] = []
    coarse_disps = None
    features_list = inputs["backbone"]['trans_features']
    for stage_idx in range(self.mvs_stages):
        features = features_list[stage_idx]
        ndc = self.num_depth_candidates_list[stage_idx]
        corr_refine_net = self.corr_refine_nets[stage_idx]
        regressor_residual = self.regressor_residuals[stage_idx]
        depth_head = self.depth_heads[stage_idx]
        b, v, c, h, w = features.shape
        h, w = int(h), int(w)
        feat_comb_lists, pose_curr_lists = prepare_feat_proj_data_lists(
            features, extrinsics)
        if stage_idx == 0:
            min_depth = rearrange(1.0 / far.clone().detach(), "b v -> (v b) 1")
            max_depth = rearrange(1.0 / near.clone().detach(), "b v -> (v b) 1")
            disp_candi_curr = (
                min_depth
                + torch.linspace(0.0, 1.0, ndc).unsqueeze(0).to(min_depth.device)
                * (max_depth - min_depth)
            ).type_as(features)
            disp_candi_curr = repeat(disp_candi_curr, "vb d -> vb d h w", h=h, w=w)
            disp_interval = (max_depth - min_depth) / (ndc - 1)
        else:
            disp_interval = disp_interval / 2
            shift = torch.linspace(-ndc // 2, ndc // 2 - 1, ndc).type_as(features)
            shift = repeat(shift, "d -> vb d h w", vb=v * b, h=h, w=w)
            disp_curr = F.interpolate(coarse_disps, scale_factor=2,
                                      mode='bilinear', align_corners=True).detach()
            disp_candi_curr = (
                disp_curr + disp_interval.view(disp_curr.shape[0], 1, 1, 1) * shift
            ).clamp(min=rearrange(min_depth, "vb 1 -> vb 1 1 1"),
                    max=rearrange(max_depth, "vb 1 -> vb 1 1 1"))
        feat01 = feat_comb_lists[0]
        raw_list = []
        for feat10, pose_curr in zip(feat_comb_lists[1:], pose_curr_lists):
            depth_vol = (1.0 / disp_candi_curr).float()
            corr = ext.cost_volume(
                feat01.half(), feat10.half(), depth_vol,
                pose_curr.float()).float()  # [vb,D,H,W]
            raw_list.append(corr)
        raw_correlation_in = torch.mean(torch.stack(raw_list, dim=0), dim=0)
        raw_correlation_in = torch.cat((raw_correlation_in, feat01), dim=1)
        if stage_idx >= 1:
            raw_correlation_in = torch.cat((raw_correlation_in, disp_curr), dim=1)
        if 'mono_depth' in inputs:
            mono_features = inputs['mono_depth']['mono_feat']
            mono_features = rearrange(mono_features, "b v c h w -> (v b) c h w")
            mono_features = F.interpolate(mono_features, size=raw_correlation_in.shape[-2:],
                                          mode='bilinear', align_corners=False)
            raw_correlation_in = torch.cat([raw_correlation_in, mono_features], dim=1)
        raw_correlation = corr_refine_net(raw_correlation_in)
        raw_correlation = raw_correlation + regressor_residual(raw_correlation_in)
        raw_correlation = depth_head(raw_correlation)
        pdf = F.softmax(raw_correlation, dim=1)
        coarse_disps = (disp_candi_curr * pdf).sum(dim=1, keepdim=True)
        pdf_max = torch.max(pdf, dim=1, keepdim=True)[0]
        outputs_stage = {}
        outputs_stage["features"] = features
        outputs_stage["disp"] = rearrange(coarse_disps, "(v b) 1 h w -> b v h w", v=v)
        outputs_stage["depth"] = 1 / outputs_stage["disp"]
        outputs_stage["photometric_confidence"] = pdf_max
        outputs_stage["raw_correlation"] = rearrange(raw_correlation, "(v b) ... -> b v ...", v=v)
        outputs_stage["disp_candi_curr"] = rearrange(disp_candi_curr, "(v b) ... -> b v ...", v=v)
        outputs["stages"].append(outputs_stage)
    outputs.update(outputs_stage)
    for stage_idx in range(self.mvs_stages, len(features_list)):
        outputs_stage = {}
        outputs_stage['features'] = features_list[stage_idx]
        outputs["stages"].append(outputs_stage)
    return outputs


def main():
    model, context = build(sys.argv[1:])
    enc = model.encoder
    with torch.inference_mode():
        gA = enc(context, inference=True)["gaussians"]["gaussians"]
    base = {k: getattr(gA, k).clone() for k in
            ("means", "covariances", "harmonics", "opacities")}
    enc.depth_predictor.forward = types.MethodType(cv_loop_cuda, enc.depth_predictor)
    with torch.inference_mode():
        gB = enc(context, inference=True)["gaussians"]["gaussians"]
    kern = {k: getattr(gB, k).clone() for k in
            ("means", "covariances", "harmonics", "opacities")}
    ok = True
    for name in ("means", "covariances", "harmonics", "opacities"):
        a = base[name].float()
        b = kern[name].float()
        cs = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
        rel = (a - b).abs().max() / (a.abs().max() + 1e-12)
        print(f"{name:12s} cos={cs.item():.9f} max_rel={rel.item():.3e}")
        ok &= cs.item() > 0.999
    print("PASS" if ok else "FAIL")
    del gA, gB, base, kern
    torch.cuda.empty_cache()
    t_base = bench(lambda: enc(context, inference=True), warmup=2, iters=5)
    t_kern = bench(lambda: enc(context, inference=True), warmup=2, iters=5)
    print(f"encoder avg: baseline={t_base[1]:.1f}ms  kernel-CV={t_kern[1]:.1f}ms")

if __name__ == "__main__":
    main()
