from __future__ import annotations

import math
import torch


def fwht_normalized(x: torch.Tensor) -> torch.Tensor:
    """Normalized Walsh-Hadamard transform over the last dimension.

    The transform is self-inverse up to floating-point roundoff:
        fwht_normalized(fwht_normalized(x)) ~= x
    """
    n = x.shape[-1]
    if n <= 0 or (n & (n - 1)):
        raise ValueError(f"last dimension must be a positive power of two, got {n}")

    orig_dtype = x.dtype
    y = x.float()
    h = 1

    while h < n:
        y = y.reshape(*y.shape[:-1], n // (2 * h), 2, h)
        a = y[..., 0, :]
        b = y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2)
        y = y.reshape(*x.shape[:-1], n)
        h *= 2

    y = y / math.sqrt(n)
    return y.to(orig_dtype)
