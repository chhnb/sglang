import argparse
from typing import Iterable, List, Tuple

import torch

from sglang.jit_kernel.norm_scale import norm_scale_fused


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


def _layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
    return (x - mean) * torch.rsqrt(var + eps) * weight + bias


def _time_cuda(fn, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def _bench_case(
    M: int,
    N: int,
    dtype: torch.dtype,
    norm_type: str,
    eps: float,
    add_const: float,
    iters: int,
) -> Tuple[float, float]:
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    scale = torch.randn(1, N, device="cuda", dtype=dtype)
    weight = torch.randn(N, device="cuda", dtype=dtype)
    bias = torch.randn(N, device="cuda", dtype=dtype)

    def run_naive():
        if norm_type == "rms":
            out = _rms_norm(x, weight, eps)
        else:
            out = _layer_norm(x, weight, bias, eps)
        out = out * (scale + add_const)
        return out

    def run_fused():
        return norm_scale_fused(
            x,
            scale,
            weight,
            bias if norm_type == "layer" else None,
            add_const=add_const,
            eps=eps,
            norm_type=norm_type,
        )

    # Warmup
    for _ in range(2):
        run_naive()
        run_fused()
    torch.cuda.synchronize()

    t_naive = _time_cuda(run_naive, iters)
    t_fused = _time_cuda(run_fused, iters)
    return t_naive, t_fused


def _parse_shape_list(shape_list: str | None) -> List[Tuple[int, int]] | None:
    if not shape_list:
        return None
    shapes: List[Tuple[int, int]] = []
    for item in shape_list.split(","):
        item = item.strip()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(f"Invalid shape '{item}', expected MxN.")
        m_str, n_str = item.lower().split("x", 1)
        shapes.append((int(m_str), int(n_str)))
    return shapes


def _iter_shapes(
    shapes: Iterable[Tuple[int, int]] | None,
) -> Iterable[Tuple[int, int]]:
    if shapes is not None:
        return shapes
    return [
        (128, 1024),
        (128, 3072),
        (128, 4096),
        (1024, 1024),
        (1024, 3072),
        (3648, 3840),
        (1024, 4096),
        (4096, 1024),
        (4096, 3072),
        (4096, 4096),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--norm-type", choices=["layer", "rms"], default="rms")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--add-const", type=float, default=1.0)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument(
        "--shapes",
        type=str,
        default=None,
        help="Comma-separated list like '3648x3840,128x1024'.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    shapes = _parse_shape_list(args.shapes)
    for M, N in _iter_shapes(shapes):
        t_naive, t_fused = _bench_case(
            M=M,
            N=N,
            dtype=dtype,
            norm_type=args.norm_type,
            eps=args.eps,
            add_const=args.add_const,
            iters=args.iters,
        )
        speedup = t_naive / t_fused if t_fused > 0 else float("inf")
        print(
            f"M={M:5d}, N={N:5d} | Naive: {t_naive:8.2f} us | "
            f"SGLang: {t_fused:8.2f} us | Speedup: {speedup:5.2f}x"
        )


if __name__ == "__main__":
    main()
