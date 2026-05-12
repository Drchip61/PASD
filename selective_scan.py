import torch
import torch.nn.functional as F


def _expand_group_param(param: torch.Tensor, dim: int) -> torch.Tensor:
    if param.dim() == 3:
        return param.unsqueeze(1).expand(-1, dim, -1, -1)
    if param.dim() == 4:
        batch, groups, state, length = param.shape
        if dim % groups != 0:
            raise ValueError(f"Cannot expand grouped selective scan params: dim={dim}, groups={groups}")
        repeat = dim // groups
        return (
            param.unsqueeze(2)
            .expand(batch, groups, repeat, state, length)
            .reshape(batch, dim, state, length)
        )
    raise ValueError(f"Unsupported param rank for selective scan: {param.dim()}")


def _selective_scan_core(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = False,
):
    input_dtype = u.dtype
    u = u.float()
    delta = delta.float()
    A = A.float()
    B = B.float()
    C = C.float()
    D = None if D is None else D.float()
    delta_bias = None if delta_bias is None else delta_bias.float()

    batch, dim, length = u.shape
    state = A.shape[1]

    if delta_bias is not None:
        delta = delta + delta_bias.view(1, -1, 1)
    if delta_softplus:
        delta = F.softplus(delta)

    B_expanded = _expand_group_param(B, dim)
    C_expanded = _expand_group_param(C, dim)

    hidden = u.new_zeros(batch, dim, state)
    outputs = []

    for t in range(length):
        dt = delta[:, :, t]
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
        dB = dt.unsqueeze(-1) * B_expanded[:, :, :, t]
        hidden = dA * hidden + dB * u[:, :, t].unsqueeze(-1)
        y_t = (hidden * C_expanded[:, :, :, t]).sum(-1)
        if D is not None:
            y_t = y_t + u[:, :, t] * D.view(1, -1)
        outputs.append(y_t)

    y = torch.stack(outputs, dim=-1).to(input_dtype)
    return y, hidden.to(input_dtype)


def selective_scan_fn(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
):
    y, last_state = _selective_scan_core(
        u=u,
        delta=delta,
        A=A,
        B=B,
        C=C,
        D=D,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
    )
    if z is not None:
        y = y * F.silu(z.to(y.dtype))
    if return_last_state:
        return y, last_state
    return y


def selective_scan_ref(*args, **kwargs):
    return selective_scan_fn(*args, **kwargs)
