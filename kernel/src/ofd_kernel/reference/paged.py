from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from .quant import quantize_int4_packed_groupwise


@dataclass
class PagedInt4KV:
    """CPU/reference representation of the Triton paged INT4 contract.

    Shapes:
      k_packed/v_packed: [P, Hkv, PAGE, G, GROUP_SIZE/2]
      k_scale/v_scale:   [P, Hkv, PAGE, G]
      block_table:       [B, max_logical_pages] (int64 on CPU reference)
      seq_lens:          [B]
    """

    k_packed: torch.Tensor
    k_scale: torch.Tensor
    v_packed: torch.Tensor
    v_scale: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    page_size: int
    group_size: int
    head_dim: int

    @property
    def num_pages(self) -> int:
        return int(self.k_packed.shape[0])

    @property
    def num_kv_heads(self) -> int:
        return int(self.k_packed.shape[1])

    @property
    def groups_per_head(self) -> int:
        return self.head_dim // self.group_size


def _validate_dense_inputs(
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
    group_size: int,
) -> tuple[int, int, int, int]:
    if k.ndim != 4 or v.ndim != 4:
        raise ValueError("k/v must have shape [B,Hkv,T,D]")
    if tuple(k.shape) != tuple(v.shape):
        raise ValueError("k and v must have identical shape")
    b, hkv, t, d = map(int, k.shape)
    if page_size <= 0 or group_size <= 0:
        raise ValueError("page_size and group_size must be positive")
    if d % group_size != 0 or group_size % 2 != 0:
        raise ValueError("head_dim must be divisible by an even group_size")
    if seq_lens.shape != (b,):
        raise ValueError(f"seq_lens must have shape [{b}]")
    if bool((seq_lens <= 0).any()) or bool((seq_lens > t).any()):
        raise ValueError("each seq_len must be in [1, T]")
    return b, hkv, t, d


def pack_dense_to_paged_int4(
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lens: torch.Tensor | None = None,
    *,
    page_size: int = 16,
    group_size: int = 32,
    physical_permutation: torch.Tensor | None = None,
    seed: int = 1337,
) -> PagedInt4KV:
    """Quantize dense logical K/V and scatter them into physical pages.

    This is a CPU/reference constructor. It intentionally randomizes physical
    page placement by default so tests exercise block-table address arithmetic
    instead of accidentally relying on logical==physical page order.
    """
    if k.device.type != "cpu" or v.device.type != "cpu":
        raise ValueError("CPU reference packer expects CPU tensors")

    if seq_lens is None:
        seq_lens = torch.full((k.shape[0],), k.shape[2], dtype=torch.long)
    seq_lens = seq_lens.to(dtype=torch.long, device="cpu")
    b, hkv, _, d = _validate_dense_inputs(
        k, v, seq_lens, page_size, group_size
    )

    pages_per_req = [math.ceil(int(n) / page_size) for n in seq_lens.tolist()]
    total_pages = sum(pages_per_req)
    max_pages = max(pages_per_req)

    if physical_permutation is None:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        physical_permutation = torch.randperm(total_pages, generator=g)
    else:
        physical_permutation = physical_permutation.to(dtype=torch.long, device="cpu")
        if physical_permutation.numel() != total_pages:
            raise ValueError("physical_permutation must cover exactly total_pages")
        if sorted(physical_permutation.tolist()) != list(range(total_pages)):
            raise ValueError("physical_permutation must be a permutation of [0,total_pages)")

    groups = d // group_size
    bytes_per_group = group_size // 2
    packed_shape = (total_pages, hkv, page_size, groups, bytes_per_group)
    scale_shape = (total_pages, hkv, page_size, groups)

    k_packed = torch.zeros(packed_shape, dtype=torch.uint8)
    v_packed = torch.zeros_like(k_packed)
    k_scale = torch.ones(scale_shape, dtype=torch.float16)
    v_scale = torch.ones_like(k_scale)
    block_table = torch.full((b, max_pages), -1, dtype=torch.long)

    logical_global = 0
    for bi, n_tokens in enumerate(seq_lens.tolist()):
        n_pages = pages_per_req[bi]
        for lp in range(n_pages):
            phys = int(physical_permutation[logical_global].item())
            logical_global += 1
            block_table[bi, lp] = phys

            start = lp * page_size
            end = min(start + page_size, int(n_tokens))
            valid = end - start

            # [Hkv, PAGE, D]; padding stays zero and is never read for valid seq_len.
            k_page = torch.zeros((hkv, page_size, d), dtype=k.dtype)
            v_page = torch.zeros_like(k_page)
            k_page[:, :valid] = k[bi, :, start:end]
            v_page[:, :valid] = v[bi, :, start:end]

            kq = quantize_int4_packed_groupwise(k_page, group_size)
            vq = quantize_int4_packed_groupwise(v_page, group_size)

            k_packed[phys] = kq.packed.reshape(hkv, page_size, groups, bytes_per_group)
            v_packed[phys] = vq.packed.reshape(hkv, page_size, groups, bytes_per_group)
            k_scale[phys] = kq.scale.reshape(hkv, page_size, groups)
            v_scale[phys] = vq.scale.reshape(hkv, page_size, groups)

    return PagedInt4KV(
        k_packed=k_packed.contiguous(),
        k_scale=k_scale.contiguous(),
        v_packed=v_packed.contiguous(),
        v_scale=v_scale.contiguous(),
        block_table=block_table.contiguous(),
        seq_lens=seq_lens.contiguous(),
        page_size=page_size,
        group_size=group_size,
        head_dim=d,
    )


def _decode_token_from_page(
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    physical_page: int,
    kv_head: int,
    page_offset: int,
    group_size: int,
    head_dim: int,
) -> torch.Tensor:
    """Decode one [D] vector using the exact byte/nibble addressing contract."""
    groups = head_dim // group_size
    out = torch.empty(head_dim, dtype=torch.float32)

    for g in range(groups):
        raw = packed[physical_page, kv_head, page_offset, g]
        lo = raw & 0x0F
        hi = (raw >> 4) & 0x0F
        code = torch.stack((lo, hi), dim=-1).reshape(group_size)
        q = code.to(torch.int16) - 8
        s = scale[physical_page, kv_head, page_offset, g].float()
        out[g * group_size : (g + 1) * group_size] = q.float() * s
    return out


def materialize_logical_cache(cache: PagedInt4KV) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize logical dense K/V for CPU oracle/debugging only.

    Returns padded [B,Hkv,Tmax,D]; callers must respect cache.seq_lens.
    """
    b = int(cache.block_table.shape[0])
    hkv = cache.num_kv_heads
    tmax = int(cache.seq_lens.max().item())
    d = cache.head_dim

    k = torch.zeros((b, hkv, tmax, d), dtype=torch.float32)
    v = torch.zeros_like(k)

    for bi in range(b):
        n = int(cache.seq_lens[bi].item())
        for tok in range(n):
            lp = tok // cache.page_size
            off = tok % cache.page_size
            phys = int(cache.block_table[bi, lp].item())
            if phys < 0:
                raise RuntimeError("valid logical token mapped to invalid physical page")
            for h in range(hkv):
                k[bi, h, tok] = _decode_token_from_page(
                    cache.k_packed,
                    cache.k_scale,
                    physical_page=phys,
                    kv_head=h,
                    page_offset=off,
                    group_size=cache.group_size,
                    head_dim=d,
                )
                v[bi, h, tok] = _decode_token_from_page(
                    cache.v_packed,
                    cache.v_scale,
                    physical_page=phys,
                    kv_head=h,
                    page_offset=off,
                    group_size=cache.group_size,
                    head_dim=d,
                )
    return k, v
