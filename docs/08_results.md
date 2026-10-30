# 08 — Results & boundaries

Statement types: measured numbers are FACT (this config, this device/versions); root-cause attributions are INTERPRETATION (experiment-backed); anything unverified would be HYPOTHESIS and is not asserted as fact here.

## Milestone chain (Orin, chain-only latency)

30,000 ms (torch eager) → 1,693 (9-graph file chain + Rodrigues) → 1,007 (GPU-resident runner + adapter kernel) → 645 (CV kernel inline) → 587.8 (gh01 split) → **558.9 (flash attention splits) ms**. Rendering adds 20.6 ms → e2e ~567 ms. Ratios against the eager baseline: 53.7× chain caliber (30,000/558.9), 52.9× end-to-end (30,000/567).

Each step was adopted on measured wins after equivalence held; each rejection also had a reason (INT8: three documented strikes — see below).

## Platform table

| Platform | Recon (chain) | Render | Notes |
|---|---:|---:|---|
| Orin AGX 64GB (sm_87, TRT 10.3) | 546.3–551.2 ms | 20.6 ms | quiet window; VLA co-resident labeled |
| A6000 (sm_86, TRT 10.16) | 129.97 ms | — | shared card ±10% |
| 4090 Laptop (sm_89, TRT 10.4) | (dev host, equivalence only) | — | not a latency contract |

## Release verification (fresh tree, 2026-10)

The released tree re-runs the full chain on the Orin and the 4090 dev host. On Orin (quiet cool window, MAXN, GR3D at 1300.5/1300.5 MHz, 99% utilization during measurement): reconstruction min-of-50 = 555.1 ms (mean 556.4), repeat runs 555.1–561.0 ms — about 1% above the quiet-window contract band above. INTERPRETATION: rebuilt engines drift ±1–2% in TRT tactic selection, consistent with the 558.9 flash-splits milestone re-measured with current source. FACT: the 2026-10 re-run passes the accuracy gate on all three platforms. On the 4090 dev host (TRT 10.4 engines, golden anchor inputs) the full gate passes: means 0.9999996, cov 0.9999764, harm 0.9998420, opa 0.9999012, render PSNR 53.15 dB (floor 52.0), chain 147.2 ms min. On the A6000 (TRT 10.16 engines) the clean-room golden tier passes with means 0.9999997, cov 0.9999824, harm 0.9998382, opa 0.9998999, render PSNR 53.22 dB - see [docs/02_pipeline.md](02_pipeline.md) for the tier semantics.

## Kernel wins (with calibers)

| Kernel | Bench caliber | In-chain caliber |
|---|---|---|
| CV | 364.3→24.5 ms (14.8×) | 364.3→43.7 ms |
| Adapter | 818→25 ms (32×) | included |
| Flash | predicted −41, measured −29 ms | segment-table audit followed |
| Rodrigues/unroll fix | 29.1 s→1.69 s (17.7×) | encoder FLOPs unchanged; compiler pathology, not math |

Bench caliber = standalone harness; in-chain = runner segment table. The two calibers differ because the bench isolates the workload; both are reported to keep honesty about where the win lands.

## INT8: three documented strikes (boundary, not failure)

Final delivery is FP16. INT8 PTQ strikes:
1. Bandwidth-bound segments: little to gain (1.11×)
2. Heavy-tail activations collapse PSNR (19.82 dB)
3. Transfer-bound segments: no gain (1.02×)

INTERPRETATION: each strike produced a reusable discriminator (where quantization helps vs not); archived with reopen conditions. FP16 with kernel replacement was the right operating point.

## Honest boundaries

- Renderer is a fork (vendored, direct-called), not a self-built rasterizer; no interactive viewer.
- Acceptance = golden single-scene equivalence; no multi-scene quality matrix.
- Cold start 3.1–3.6 s (untreated), separate from steady state.
- 256-res tier: latency measured (A6000 ~79 ms), quality never evaluated — decision data missing, frozen.
- INT8: boundary discovery, not success.
