# 04 — Custom CUDA kernels

Five custom kernel classes (41 launches in the chain) replace the segments where TensorRT's compiler systematically failed. Each one was adopted only after a bench or equivalence harness proved it out; none is hand-tuned assembly — straightforward CUDA, chosen because the fused-graph alternative was pathologically slow.

## The five classes

| Kernel class | Files | Replaces | Win |
|---|---|---|---|
| Cost volume | [cxx/cost_volume_bench.cu](../cxx/cost_volume_bench.cu) (bench), [cxx/cv_kernel_torch.cu](../cxx/cv_kernel_torch.cu) (torch cross-check), inlined in [runner.cu](../cxx/runner.cu) | grid_sample+correlation pointwise chain — 70× off the bandwidth roofline | bench 364.3→24.5 ms (14.8×); in-chain 43.7 ms |
| Adapter chain | [cxx/ada_chain_torch.cu](../cxx/ada_chain_torch.cu) (torch cross-check) + inlined in [runner.cu](../cxx/runner.cu) | fp32 K=3 GEMMs forced onto FFMA+trig chains | 818→25 ms (32×) |
| Flash attention | [cxx/flash_attn.cu](../cxx/flash_attn.cu) | materializing 16384² fp16 attention scores (537 MB per block, DRAM×4) | −29 ms chain-wide |
| Boundary casts | in [cxx/runner.cu](../cxx/runner.cu) | scatter of dtype mismatches at engine boundaries | one mechanism, `cast_f16_f32` / `cast_f32_f16` |
| Render bench | [cxx/render_bench.cu](../cxx/render_bench.cu) + [cxx/render_matrix.cu](../cxx/render_matrix.cu) | torch-bound rasterizer calls | 20.6 ms torch-free, bit-identical across three platforms |

## Cost volume

The upstream cost volume builds grid_sample + correlation as a pointwise graph — a fused TRT graph ran 70× off the memory-bandwidth roofline. The custom kernel is one pass: read features, build the correlation volume directly. Roofline analysis came first: the fused graph was never compute bound, so no quantization route could fix it. Sources include a torch cross-check ([cv_kernel_torch.cu](../cxx/cv_kernel_torch.cu)) and a standalone bench ([cost_volume_bench.cu](../cxx/cost_volume_bench.cu)).

## Adapter chain (ada)

Each gaussian-head stage ends in an adapter: small GEMMs (K=3) + quaternion → rotation + Rodrigues SH rotation. In the fused TRT graph these fp32 small-GEMM chains landed on FFMA + trig instruction chains. The custom kernel keeps the chain in registers: `apply_e`, `quat_to_R`, `ada_kernel`. Torch cross-check: [cxx/ada_chain_torch.cu](../cxx/ada_chain_torch.cu) + [scripts/test_ada_kernel.py](../scripts/test_ada_kernel.py).

## Flash attention

Two attention segments (cv_post1: H=2/S=4096/D=32; cv_post2: H=1/S=16384/D=32) materialized attention scores at 16384² fp16 — 537 MB per block, 4× DRAM round-trips. The custom flash kernel (`mma.m16n8k16` tensor-core fragments, online softmax) avoids materialization. Splitting: each graph's 3 attention blocks → a/m1/m2/b engine segments around the kernel boundary. Predicted −41 ms, measured −29 ms — the gap drove the segment table audit.

**Gotcha**: the mma m16n8k16 A-fragment layout maps (g+8, 2c) — easy to get wrong silently; probe with the mma_unit harness before trusting the kernel. ([cxx/flash_attn.cu](../cxx/flash_attn.cu) includes unit probes.)

## Boundary casts

Engine boundaries converge dtype at exactly two kernels (`cast_f16_f32_kernel`, `cast_f32_f16_kernel` in [runner.cu](../cxx/runner.cu)). fp16 in, fp32 math.

## Render bench

[render_bench.cu](../cxx/render_bench.cu) calls the vendored rasterizer (`CudaRasterizer::Rasterizer::forward`) directly — no torch, no python. 6×256² cubemap → 512×1024 ERP pano in 20.6 ms on Orin, bit-identical across three platforms (sm_86/87/89, same source, only `-arch` differs). Findings: render is blend-bound; tile-count parallelism floor — 128² tiles are *slower* than 256² (fewer CTAs than SMs can use).
