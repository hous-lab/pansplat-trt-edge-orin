# Vendored components

This directory holds third-party code vendored into the tree so a fresh clone
builds without fetching submodules. Each component keeps its original license
verbatim; the repository's MIT license does not extend to them.

## diff-gaussian-rasterization-modified

- **Upstream**: https://github.com/dcharatan/diff-gaussian-rasterization-modified
- **Snapshot**: upstream commit `1250c42` ("Note about new and improved modifications")
- **License**: the Gaussian-Splatting License (Inria/MPII, research use, see
  `diff-gaussian-rasterization-modified/LICENSE.md` inside the vendored copy —
  the rasterizer derives from graphdeco-inria/gaussian-splatting, so the
  vendored directory is **not** MIT and commercial use of the rasterizer
  component requires contacting Inria.
- **Why vendored**: the deployment runner (cxx/runner.cu) links the rasterizer
  for the rendering benchmarks (render_bench.cu / render_matrix.cu) and for the
  PyTorch-side comparison paths; a fresh clone needs it to build cxx/ targets
  that rasterize.
- **Changes**: none. The vendored copy is a plain snapshot; the upstream
  README's own note ("Spherical harmonics match e3nn now") describes the
  modifications upstream made relative to the Inria rasterizer, which is
  exactly what the 88-dim (SH degree 4) Gaussians need.
- **Build**: `pip install -e ./third_party/diff-gaussian-rasterization-modified`
  (the release pipeline's cxx step builds only the runner and bench binaries;
  the Python extension is needed for the torch-side comparison and rendering
  scripts).