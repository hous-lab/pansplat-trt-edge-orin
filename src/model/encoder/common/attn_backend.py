"""Attention backend dispatcher (deployment breakpoint B1).

xformers memory_efficient_attention cannot be exported to ONNX/TRT; this module
funnels all 3 call sites into one entry point with two backends:
  - xformers: upstream baseline behavior, numerical reference (golden)
  - sdpa:     F.scaled_dot_product_attention, mathematically equivalent, exportable
Switch via the PANSPLAT_ATTN_BACKEND=sdpa environment variable, or set_backend("sdpa").
"""

import os

import torch

import torch.nn.functional as F

try:
    from xformers.ops import memory_efficient_attention as _mea
except ImportError:  # pragma: no cover
    _mea = None

_BACKEND = os.environ.get("PANSPLAT_ATTN_BACKEND", "xformers").lower()


def get_backend() -> str:
    return _BACKEND


def set_backend(name: str) -> None:
    global _BACKEND
    _BACKEND = name.lower()
    if _BACKEND == "xformers" and _mea is None:
        raise ImportError("xformers is not installed")


def attention(q, k, v, attn_mask=None):
    """Drop-in equivalent interface of memory_efficient_attention.

    q: [B, M, C]  k, v: [B, S, C]
    attn_mask: additive float mask, broadcastable, or None (same semantics as the
    xformers attn_bias).
    Returns the same dtype as the input.
    """
    if _BACKEND == "xformers" and _mea is not None:
        # xformers needs bf16 (baseline behavior; the cast is funneled here so call sites
    # stay backend-agnostic)
        out = _mea(q.to(torch.bfloat16).contiguous(),
                   k.to(torch.bfloat16).contiguous(),
                   v.to(torch.bfloat16).contiguous(),
                   attn_bias=attn_mask.to(torch.bfloat16) if attn_mask is not None else None)
        return out.to(q.dtype)
    # sdpa: pure fp32 (exportable; in TRT FP16 mode the builder downcasts on its own)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
