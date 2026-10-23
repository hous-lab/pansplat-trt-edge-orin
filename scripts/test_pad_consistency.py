"""Module-level numerical-consistency check for the B16 fix: circular pad vs concat
wrap must be bit-identical on the same input."""
import sys
sys.path.insert(0, ".")
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

def ref_conv_wrap(x, conv, pad_h, pad_w):
    """Original implementation: F.pad (vertical-constant + horizontal-circular)."""
    y = F.pad(x, (0, 0, pad_h, pad_h), mode="constant")
    y = F.pad(y, (pad_w, pad_w, 0, 0), mode="circular")
    return conv(y)

def ref_pool_wrap(x, pool_pad_fn):
    raise NotImplementedError

# 1) Direct function-level comparison: 4D/5D inputs, circular vs concat
print("=== function-level: circular vs concat (varying pad/shape/dims) ===")
ok = True
for shape, p in [((2, 8, 16, 32), 1), ((2, 8, 16, 32), 2), ((2, 8, 16, 32), 3),
                 ((1, 3, 8, 8), 1), ((2, 4, 5, 7), 1),   # odd sizes
                 ((2, 4, 5, 7), 0),                       # p=0 (edge case)
                 ((1, 4, 3, 6, 8), 1),                    # 5D: verifies dim=-1 correctness
                 ]:
    x = torch.randn(*shape)
    # circular reference
    ref = F.pad(x, (p, p, 0, 0), mode="circular")
    # concat implementation (with p>0 guard)
    if p > 0:
        new = torch.cat([x[..., -p:], x, x[..., :p]], dim=-1)
    else:
        new = x
    d = (ref - new).abs().max().item()
    status = "✓" if d == 0 else "✗ MISMATCH"
    ok &= (d == 0)
    print(f"shape={shape} pad={p}: max_abs_diff={d:.2e} {status}")
print("function-level match:", "PASS" if ok else "FAIL")
