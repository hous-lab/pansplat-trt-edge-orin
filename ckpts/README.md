# Checkpoints and upstream artifacts

Weights used by the deployment pipeline. Upstream checkpoint files are **not
stored in this repository** and are never rehosted — this file records how to
obtain them, where to place them, and how to verify them after download.

## Download

| File | Description | sha256 | Place at |
|---|---|---|---|
| `last.ckpt` | PanSplat pretrained on Matterport3D at 512×1024 (upstream `nvpl49ge` release) | `3b060c29b70a1fbda5f96efa94a75cd1fd70f8d2a5820dd9df06a36f0f8241ee` | `logs/nvpl49ge/checkpoints/last.ckpt` |

Download link: upstream README "Quick Demo on Synthetic Data" section —
[archived copy](../docs/upstream_readme.md#-quick-demo-on-synthetic-data) (the
SharePoint URL is in the archived upstream README and in the upstream repo
README directly).

Verify after download:

```bash
sha256sum logs/nvpl49ge/checkpoints/last.ckpt
# expect 3b060c29b70a1fbda5f96efa94a75cd1fd70f8d2a5820dd9df06a36f0f8241ee
```

The pipeline guard (step 1 of [run_pipeline.sh](../run_pipeline.sh)) fails fast with this pointer
when the checkpoint is missing.

## Other upstream checkpoints (not used by this pipeline)

Upstream also publishes fine-tuned checkpoints (360Loc 512/1024/2048, 4K
Matterport3D) from the same SharePoint folder — see the upstream README
([archived copy](../docs/upstream_readme.md)). The deployment pipeline in this
repository targets the 512×1024 Matterport3D checkpoint only. Additionally,
the upstream UniMatch / PanoGRF monocular-depth pretrained weights (same
upstream README "weights" link, `checkpoints` folder) are needed only if you
re-run the upstream dataset preprocessing that produces `mono_image` /
`cube_image` inputs; the deployment chain itself needs only `last.ckpt`.

## Policy

Upstream-hosted files — weights **and** any anchor/statistics files the
upstream publishes — are download-only: fetch from the upstream source, place
at the path listed above, verify sha256. None are committed to this repository,
and rehosting them on our Hugging Face — including any retrained or derived
variant — is prohibited. The Hugging Face companion repo (https://huggingface.co/hous-lab/pansplat-trt-edge-orin)
hosts only our own pipeline-exported
artifacts (ONNX segments, prebuilt engines, golden anchor).