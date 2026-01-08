from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Literal

import torch

try:
    from sglang.jit_kernel.utils import load_jit, make_cpp_args
except ModuleNotFoundError:
    from jit_kernel.utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@functools.cache
def _jit_norm_scale_module(
    norm_type: int, scale_broadcast: bool, item_per_thread: int
) -> Module:
    args = make_cpp_args(norm_type, scale_broadcast, item_per_thread)
    return load_jit(
        "norm_scale_fused",
        *args,
        cuda_files=["norm_scale_fused.cuh"],
        cuda_wrappers=[("norm_scale_fused", f"norm_scale_fused<{args}>")],
    )


def _prepare_inputs(
    x: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    if x.dim() == 3:
        if not x.is_contiguous():
            x = x.contiguous()
        B, L, N = x.shape
        x2d = x.view(B * L, N)
    elif x.dim() == 2:
        if not x.is_contiguous():
            x = x.contiguous()
        B = 1
        L, N = x.shape
        x2d = x
    else:
        raise ValueError("x must be 2D [M,N] or 3D [B,L,N]")

    if scale.dim() == 3:
        if not scale.is_contiguous():
            scale = scale.contiguous()
        if scale.shape[1] != 1:
            raise ValueError("scale must have shape [B,1,N] or [1,1,N]")
        if scale.shape[2] != N:
            raise ValueError("scale last dim must match N")
        scale2d = scale.reshape(scale.shape[0], N)
    elif scale.dim() == 2:
        if not scale.is_contiguous():
            scale = scale.contiguous()
        if scale.shape[1] != N:
            raise ValueError("scale last dim must match N")
        scale2d = scale
    else:
        raise ValueError("scale must be 2D [B,N] or 3D [B,1,N]")

    if scale2d.shape[0] not in (1, B):
        raise ValueError("scale must have B==1 or B==batch size")

    return x2d, scale2d, B, L, N


def norm_scale_fused(
    x: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    add_const: float,
    eps: float,
    norm_type: Literal["rms", "layer"],
) -> torch.Tensor:
    x2d, scale2d, B, L, N = _prepare_inputs(x, scale)
    if not x2d.is_cuda:
        raise ValueError("norm_scale_fused only supports CUDA tensors")
    if x2d.stride(-1) != 1:
        x2d = x2d.contiguous()
    if scale2d.stride(-1) != 1:
        scale2d = scale2d.contiguous()
    if weight.stride(-1) != 1:
        weight = weight.contiguous()
    if bias is None:
        bias = weight
    elif bias.stride(-1) != 1:
        bias = bias.contiguous()

    item_per_thread = 1 if N <= 4096 else 8
    scale_broadcast = scale2d.shape[0] == 1
    norm_type_val = 0 if norm_type == "layer" else 1

    module = _jit_norm_scale_module(norm_type_val, scale_broadcast, item_per_thread)
    out2d = torch.empty_like(x2d)
    module.norm_scale_fused(
        out2d, x2d, scale2d, weight, bias, L, float(add_const), float(eps)
    )
    if x.dim() == 3:
        return out2d.view(B, L, N)
    return out2d
