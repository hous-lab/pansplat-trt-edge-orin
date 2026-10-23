#!/usr/bin/env python3
"""Synth-tier self-anchored gate: torch reference vs runner gaussian dump.

Both sides consume identical inputs (export/runner_bins/): the runner reads
the binary inputs including xy_s*.bin, while the torch reference overrides
the gaussian-head gs_xy_*_0 buffers with the same bin values before running
the chain (GHAdapterSeg clones xy at construction, so buffers are overridden
before the chain modules are built). Gate: per-field cosine >= 0.9997 -
deployment equivalence, not dataset accuracy.
"""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PRE = None
OVERRIDES = []
for a in sys.argv[1:]:
    if a.startswith(("-", "+")) or "=" in a:
        OVERRIDES.append(a)
    else:
        PRE = a
if PRE is None:
    PRE = str(REPO / "export" / "scratch" / "synth_run")
# The torch reference needs the upstream checkpoint. run_pipeline passes the
# same hydra overrides it uses for the exporters; a bare call falls back to
# the standard ckpt placement (PANSPLAT_CKPT to relocate).
if not OVERRIDES:
    import os
    ckpt = os.environ.get("PANSPLAT_CKPT", "logs/nvpl49ge/checkpoints/last.ckpt")
    OVERRIDES = ["+experiment=pansplat-512", "mode=predict",
                 f"++model.weights_path={ckpt}"]
import types
sys.modules.setdefault("wandb", types.ModuleType("wandb"))

import numpy as np
import torch
from jaxtyping import install_import_hook

with install_import_hook(("src",), ("beartype", "beartype")):
    from hydra import initialize, compose
    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper
    from src.model.encoder.common import attn_backend
    from src.misc.sh_rotation import set_sh_backend
    from src.model.encoder.gh_split import (MVSGraph, GHStageGraph,
                                            GHCnnSeg, GHMlpSeg, GHAdapterSeg)
import warnings
warnings.filterwarnings("ignore")

BINS = REPO / "export" / "runner_bins"
CTX_SHAPES = {"image": (1, 2, 3, 512, 1024),
              "mono_image": (1, 2, 3, 512, 1024),
              "cube_image": (1, 2, 3, 256, 1536),
              "extrinsics": (1, 2, 4, 4),
              "near": (1, 2), "far": (1, 2)}
XY_COUNTS = {0: 5215, 1: 20860, 2: 83443, 3: 333772}
GATE = 0.9997


def build_model(ovr):
    with initialize(config_path="../config", version_base=None):
        cfg_dict = compose(config_name="main", overrides=ovr)
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg_dict.seed)
    torch.set_float32_matmul_precision("high")
    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    m = ModelWrapper(cfg.optimizer, cfg.test, cfg.train, cfg.val, cfg.predict,
                     encoder, None, decoder, [], None,
                     cfg.model.weights_path, cfg.model.wo_defbp2)
    return m.eval().cuda()


def load_bins():
    ctx = {}
    for k, shp in CTX_SHAPES.items():
        n = int(np.prod(shp))
        v = np.fromfile(BINS / f"{k}.bin", dtype=np.float32)
        assert v.size == n, (k, v.size, n)
        ctx[k] = torch.from_numpy(v.reshape(shp)).cuda()
    xy = {}
    for s, cnt in XY_COUNTS.items():
        v = np.fromfile(BINS / f"xy_s{s}.bin", dtype=np.float32)
        assert v.size == cnt * 2, (s, v.size)
        xy[s] = torch.from_numpy(v.reshape(cnt, 2)).cuda()
    return ctx, xy


def main():
    model = build_model(OVERRIDES)
    enc = model.encoder
    ghead = enc.gaussian_head
    ctx, xy = load_bins()
    with torch.inference_mode():
        for s in XY_COUNTS:
            getattr(ghead, f"gs_xy_{s}_0").copy_(xy[s])
        mvs = MVSGraph(enc).eval().cuda()
        a_out = mvs(ctx["image"], ctx["extrinsics"], ctx["near"],
                    ctx["far"], ctx["mono_image"], ctx["cube_image"])
        feats, rc, dc = a_out[:4], a_out[4], a_out[5]
        image, extr, near, far = a_out[6:10]
        last_raw = None
        gcols = []
        for s in range(4):
            if s >= 2:
                cnn = GHCnnSeg(enc, s).eval().cuda()
                mlp = GHMlpSeg(enc, s).eval().cuda()
                ada = GHAdapterSeg(enc, s).eval().cuda()
                raw_cnn, images_c = cnn(feats[s], rc, image, last_raw)
                raw_g, dc_s, shb = mlp(raw_cnn, images_c, dc)
                outs = ada(raw_g, dc_s, shb, near, far, extr)
                gcols.append(outs[:4])
                last_raw = raw_cnn
            else:
                sg = GHStageGraph(enc, s).eval().cuda()
                ins = [feats[s], rc, dc, image, extr, near, far]
                if s > 0:
                    ins.append(last_raw)
                outs = sg(*ins)
                gcols.append(outs[:4])
                last_raw = outs[4]
        ok = True
        print(f"{'field':12s} {'max_abs':>10s} {'cosine':>12s}")
        g4 = [torch.cat([g[j].reshape(-1) for g in gcols]) for j in range(4)]
        for j, k in enumerate(("means", "cov", "harm", "opa")):
            t = g4[j].float().cpu().numpy().reshape(-1)
            r = np.fromfile(f"{PRE}.{k}.f32", dtype=np.float32)
            if r.size != t.size:
                print(k, "SIZE MISMATCH", r.size, t.size); ok = False
                continue
            c = float(np.dot(t.astype(np.float64), r.astype(np.float64))
                      / (np.linalg.norm(t.astype(np.float64))
                         * np.linalg.norm(r.astype(np.float64)) + 1e-12))
            md = float(np.abs(t - r).max())
            print(f"{k:6s} {md:10.3e} {c:12.9f}")
            ok &= c > GATE
    print("SYNTH-GATE-PASS" if ok else "SYNTH-GATE-FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    main()
