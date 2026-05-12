import torch

from selective_scan import _selective_scan_core


def fwd(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = False,
    nrows: int = 1,
):
    # Pure PyTorch inference fallback for the missing CUDA kernel.
    out, hidden = _selective_scan_core(
        u=u,
        delta=delta,
        A=A,
        B=B,
        C=C,
        D=D,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
    )
    return out, hidden


def bwd(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    delta_bias: torch.Tensor | None,
    dout: torch.Tensor,
    x: torch.Tensor,
    delta_softplus: bool,
    nrows: int,
):
    # Backward is not needed for test-time failure export. Return zeros for safety.
    du = torch.zeros_like(u)
    ddelta = torch.zeros_like(delta)
    dA = torch.zeros_like(A)
    dB = torch.zeros_like(B)
    dC = torch.zeros_like(C)
    dD = None if D is None else torch.zeros_like(D)
    ddelta_bias = None if delta_bias is None else torch.zeros_like(delta_bias)
    return du, ddelta, dA, dB, dC, dD, ddelta_bias
