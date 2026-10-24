# 06 — Export chain

[scripts/export_all.py](../scripts/export_all.py) drives five producer passes (fail-fast, per-step logs under `export/scratch/export_all/`):

1. [export_cv_split.py](../scripts/export_cv_split.py) — traces mvs_pre, cv_pre0/1/2, cv_post0 + cv_post1/2 intermediates (extract_model + boundary casts at attention boundaries)
2. [export_split_post.py](../scripts/export_split_post.py) ×2 — splits cv_post1/cv_post2 around their 3 attention blocks each: a/m1/m2/b segments (fp16 boundary casts; --no-cast for bit-exact verify mode)
3. [export_gh_s01.py](../scripts/export_gh_s01.py) — traces gh_s{0,1}_{cnn,mlp}
4. `export_fine.py --only=<graph>` ×4 — gh_s{2,3}_{cnn,mlp}

All torch-side producers take hydra overrides (`+experiment=pansplat-512 mode=predict ++model.weights_path=...`; mode=predict keeps the unimatch weights loader out of the path). All trace with `export/golden_512.npz` as the input scene — no dataset needed at export time.

## Split mechanics ([export_split_post.py](../scripts/export_split_post.py))

- Find attention blocks by node-name suffix (`/attention/Transpose_1`)
- Back-reach from each cut tensor to find each segment's node set
- Boundary casts fp32↔fp16 at each cut (engine inputs fp16, math fp32)
- 0-dim boundary shapes fixed via a one-shot ORT oracle
- Writes p{1,2}_{a,m1,m2,b}.onnx + split_meta.json

## Size budget

Trace once, split once: sub-graph extraction reaches back into the same initializers, so the 23 ONNX graphs share init tensors — total ~160 MB. Re-tracing each segment separately would re-materialize the backbone weights per segment (~1 GB). The rule: trace once, split, don't re-trace.
