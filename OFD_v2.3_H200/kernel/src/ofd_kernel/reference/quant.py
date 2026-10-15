from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class PackedInt4:
    packed: torch.Tensor
    scale: torch.Tensor
    orig_shape: tuple[int, ...]
    group_size: int

    @property
    def storage_bytes(self) -> int:
        return (
            self.packed.numel() * self.packed.element_size()
            + self.scale.numel() * self.scale.element_size()
        )


def quantize_int4_packed_groupwise(
    x: torch.Tensor,
    group_size: int = 32,
) -> PackedInt4:
    """Exact packing contract used by the original OFD prototype.

    The 4-bit storage can represent [-8, 7] and is encoded as q + 8 in
    uint4, low-nibble first. With the symmetric absmax/7 scale used here,
    ordinary finite inputs quantize to [-7, 7]; -8 remains representable by
    the storage contract but is normally unused by this quantizer.
    """
    if x.shape[-1] % group_size != 0:
        raise ValueError("last dimension must be divisible by group_size")
    if group_size % 2 != 0:
        raise ValueError("group_size must be even")

    orig_shape = tuple(x.shape)
    d = orig_shape[-1]
    g = d // group_size

    xf = x.float().reshape(-1, g, group_size)
    scale = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7.0
    q = torch.round(xf / scale).clamp(-8, 7).to(torch.int16)
    code = (q + 8).to(torch.uint8)

    lo = code[..., 0::2]
    hi = code[..., 1::2]
    packed = (lo | (hi << 4)).contiguous()

    return PackedInt4(
        packed=packed,
        scale=scale.squeeze(-1).to(torch.float16).contiguous(),
        orig_shape=orig_shape,
        group_size=group_size,
    )


def dequantize_int4_packed_groupwise(
    qobj: PackedInt4,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    packed = qobj.packed
    scale = qobj.scale.float()

    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F

    code = torch.stack((lo, hi), dim=-1).reshape(
        *packed.shape[:-1],
        qobj.group_size,
    )
    q = code.to(torch.int16) - 8
    x = q.float() * scale.unsqueeze(-1)
    return x.reshape(qobj.orig_shape).to(out_dtype)


def qdq_int4_groupwise(
    x: torch.Tensor,
    group_size: int = 32,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    q = quantize_int4_packed_groupwise(x, group_size)
    return dequantize_int4_packed_groupwise(
        q,
        out_dtype=x.dtype if out_dtype is None else out_dtype,
    )
