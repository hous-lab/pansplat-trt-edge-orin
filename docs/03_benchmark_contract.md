# 03 — Benchmark contract

All published numbers come from the same measurement discipline. Numbers measured differently are not comparable.

## Platform

Jetson AGX Orin 64GB, JetPack 6.2, TensorRT 10.3, MAXN with live clock readback (GR3D 1300.5 MHz read at measurement time). Quiet window: no other GPU process; co-resident numbers are labeled (±10%).

## Rules

1. **Quiet window**, or label ±10% co-resident.
2. **MAXN + clock readback** — numbers without clock readback are not comparable to this contract.
3. **min-of-N** (N ≥ 3), report min and mean.
4. **What is timed**: reconstruction = chain start → concat done. Rendering timed separately (torch-free bench binary). e2e = the two summed.
5. H2D/D2H measured separately (~6–13 ms); not in the e2e figure.
6. Same-environment A/B only: cross-environment PSNR comparisons are invalid (observed 120→14 dB drift once across environments).
7. Report the engine platform dir (arch+TRT version) alongside numbers.

## Reference numbers

| Metric | Value | Scope |
|---|---|---|
| Reconstruction (chain) | 546.3–549.4 ms | min / 5-run mean 548.7 ms, quiet window, MAXN |
| + co-resident | 551.2 ms | ±10% co-resident window |
| Rendering | 20.6 ms | 6×256² cubemap → ERP 512×1024, torch-free bench |
| End-to-end (recon + render) | ~567 ms | recon + render summed |
| vs PyTorch eager on Orin | 53.7× | 30,000 ms eager chain |
| Gaussian fields cosine | ≥ 0.99978 | 4 fields vs torch golden |
| Render PSNR | 52.95 dB | vs torch golden render (52.96 dB) |
| A6000 (sm_86, TRT 10.16) | 129.97 ms | shared card, ±10% |
| Cold start | 3.1–3.6 s | to first frame, separate from steady state |

## Platform pairing

Engines are keyed by arch+TRT version (`export/engines/<arch>-trt<version>/`). Numbers are only comparable within the same pairing. Orin = sm_87 + TRT 10.3. A6000 = sm_86 + TRT 10.16. The dev-host (4090 Laptop, sm_89, TRT 10.3) is used for equivalence cross-checks, not latency contracts.
