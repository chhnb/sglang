import pathlib
import sys

import torch

root = pathlib.Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from jit_kernel.norm_scale import norm_scale_fused


def _rms_ref(x, weight, eps):
    x_f = x.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    y = x_f * rstd
    y = y * weight
    return y.to(x.dtype)


def _ln_ref(x, weight, bias, eps):
    x_f = x.float()
    mean = x_f.mean(dim=-1, keepdim=True)
    var = (x_f - mean).pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    y = (x_f - mean) * rstd
    y = y * weight + bias
    return y.to(x.dtype)


def _run_case(norm_type: str):
    torch.manual_seed(0)
    B, L, N = 2, 3, 128
    eps = 1e-6
    add_const = 1.0

    x = torch.randn(B, L, N, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(B, 1, N, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(N, device="cuda", dtype=torch.bfloat16)

    out = norm_scale_fused(
        x,
        scale,
        weight,
        bias if norm_type == "layer" else None,
        add_const=add_const,
        eps=eps,
        norm_type=norm_type,
    )
    if norm_type == "rms":
        base = _rms_ref(x, weight, eps)
    else:
        base = _ln_ref(x, weight, bias, eps)
    ref = (base.float() * (scale.float() + add_const)).to(x.dtype)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

    # Broadcast scale [1, 1, N]
    scale_b = scale[:1].contiguous()
    out_b = norm_scale_fused(
        x,
        scale_b,
        weight,
        bias if norm_type == "layer" else None,
        add_const=add_const,
        eps=eps,
        norm_type=norm_type,
    )
    ref_b = (base.float() * (scale_b.float() + add_const)).to(x.dtype)
    torch.testing.assert_close(out_b, ref_b, rtol=1e-2, atol=1e-2)
    print(f"{norm_type} is Ok")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; skipping.")
        return
    _run_case("rms")
    _run_case("layer")


if __name__ == "__main__":
    main()
