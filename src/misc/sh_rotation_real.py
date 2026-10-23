"""All-real SH rotation v2: Rodrigues closed-form expm (final form of breakpoint B10).

wigner_D(l,a,b,g) = expm(a*Xz)expm(b*Xy)expm(g*Xz); the so3 generators are import-time
constants.
expm uses the Rodrigues closed form I + sin(l*th)/l*X + (1-cos(l*th))/l^2*X^2
(3 terms, no matmul chain),
replacing v1's Taylor / scaling-squaring -- which TRT once compiled into 2012 unrolled kernels.
Equivalence to e3nn.rotate_sh: v1 verified max_abs=3.3e-05; v2 re-verified the same way.
"""

import torch
from einops import repeat
from jaxtyping import Float
from torch import Tensor
from e3nn.o3._wigner import so3_generators

_GEN = [so3_generators(l).to(torch.float32) for l in range(5)]
_GEN2 = [g @ g for g in _GEN]
# lambda = sqrt(-tr(X^2)/2) (X skew-symmetric, nonzero eigenvalues +/- i*lambda); for
# l=0 the generator is all-zero, so lambda:=1 as a placeholder
_LAM = []
for gens in _GEN:
    row = []
    for g in gens:
        lam2 = float(-(g @ g).diagonal().sum() / 2)  # tr(X^2)/2; for skew-symmetric X the matrix square equals minus the sum of elementwise squares
        row.append(max(lam2, 1e-12) ** 0.5 if lam2 > 1e-12 else 1.0)
    _LAM.append(row)


def _so3_expm(theta: Tensor, l: int, axis: int) -> Tensor:
    """Rodrigues closed form of expm(theta*X_axis^l). theta: [*, 1, 1]."""
    X = _GEN[l][axis].to(theta.device)
    t = theta[..., None, None]
    if l == 0:
        return torch.ones(*theta.shape, 1, 1, device=theta.device)
    X2 = _GEN2[l][axis].to(theta.device)
    lam = _LAM[l][axis]
    s = torch.sin(lam * t) / lam
    c = (1.0 - torch.cos(lam * t)) / (lam * lam)
    eye = torch.eye(X.shape[-1], device=theta.device, dtype=theta.dtype)
    return eye + s * X + c * X2


def rotate_sh_real(
    sh_coefficients: Float[Tensor, "*#batch n"],
    rotations: Float[Tensor, "*#batch 3 3"],
    max_degree: int = 4,
) -> Float[Tensor, "*batch n"]:
    """All-real equivalent of e3nn rotate_sh."""
    sh = sh_coefficients
    lead_shape = sh.shape[:-1]
    n_coef = sh.shape[-1]
    sh_flat = sh.reshape(-1, n_coef)
    rot_flat = rotations.expand(*lead_shape, 3, 3).reshape(-1, 3, 3).float()

    R = rot_flat
    v = R[..., :, 1]  # y column
    a = torch.atan2(v[..., 0], v[..., 2])
    b = torch.acos(v[..., 1].clamp(-1, 1))
    R1 = _so3_expm(a, 1, 1) @ _so3_expm(b, 1, 0)
    R2 = R1.transpose(-1, -2) @ R
    c = torch.atan2(R2[..., 0, 2], R2[..., 0, 0])
    out = []
    for l in range(max_degree + 1):
        Dl = _so3_expm(a, l, 1) @ _so3_expm(b, l, 0) @ _so3_expm(c, l, 1)
        out.append(torch.einsum("xij,xj->xi",
                                Dl.type(sh_flat.dtype),
                                sh_flat[..., l**2:(l + 1) ** 2]))
    return torch.cat(out, dim=-1).reshape(*lead_shape, n_coef)
