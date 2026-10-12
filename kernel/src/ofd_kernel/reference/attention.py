from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from ..contracts import Variant
from .fwht import fwht_normalized
from .paged import PagedInt4KV, materialize_logical_cache, pack_dense_to_paged_int4
from .quant import qdq_int4_groupwise


def _expand_gqa(
    k: torch.Tensor,
    v: torch.Tensor,
    num_q_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand [B,Hkv,T,D] to query-head view for CPU/reference use only."""
    hkv = k.shape[1]
    if v.shape[1] != hkv:
        raise ValueError("K/V head counts must match")
    if num_q_heads % hkv != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    repeat = num_q_heads // hkv
    return (
        k.repeat_interleave(repeat, dim=1),
        v.repeat_interleave(repeat, dim=1),
    )


def _validate_seq_lens(seq_lens: torch.Tensor, batch: int, tmax: int) -> torch.Tensor:
    seq_lens = seq_lens.to(dtype=torch.long, device="cpu")
    if seq_lens.shape != (batch,):
        raise ValueError(f"seq_lens must have shape [{batch}]")
    if bool((seq_lens <= 0).any()) or bool((seq_lens > tmax).any()):
        raise ValueError("each seq_len must be in [1, T]")
    return seq_lens


def attention_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference decode attention with optional ragged sequence lengths.

    q: [B,Hq,D]
    k/v: [B,Hkv,T,D]
    returns: [B,Hq,D]
    """
    kq, vq = _expand_gqa(k, v, q.shape[1])
    b, _, t, _ = kq.shape
    scores = torch.einsum("bhd,bhtd->bht", q.float(), kq.float())
    scores = scores / math.sqrt(q.shape[-1])

    if seq_lens is not None:
        lens = _validate_seq_lens(seq_lens, b, t).to(scores.device)
        tok = torch.arange(t, device=scores.device)
        valid = tok[None, None, :] < lens[:, None, None]
        scores = scores.masked_fill(~valid, -float("inf"))

    p = torch.softmax(scores, dim=-1)
    out = torch.einsum("bht,bhtd->bhd", p, vq.float())
    return out.to(q.dtype)


@dataclass
class SplitState:
    m: torch.Tensor  # [S,B,H]
    l: torch.Tensor  # [S,B,H]
    o: torch.Tensor  # [S,B,H,D]


def split_kv_partials(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    split_size: int,
    *,
    seq_lens: torch.Tensor | None = None,
    num_splits: int | None = None,
) -> SplitState:
    """Produce stable split-KV partial states, including empty splits.

    Empty splits use the explicit neutral state:
        m=-inf, l=0, o=0
    so ragged batches never evaluate an effective `-inf - (-inf)` merge.
    """
    if split_size <= 0:
        raise ValueError("split_size must be positive")

    kq, vq = _expand_gqa(k, v, q.shape[1])
    b, h, t, d = kq.shape
    if seq_lens is None:
        seq_lens = torch.full((b,), t, dtype=torch.long)
    seq_lens = _validate_seq_lens(seq_lens, b, t)

    required = math.ceil(int(seq_lens.max().item()) / split_size)
    if num_splits is None:
        num_splits = required
    if num_splits < required:
        raise ValueError("num_splits is too small for max(seq_lens)")

    scale = 1.0 / math.sqrt(d)
    m_all = torch.full((num_splits, b, h), -float("inf"), dtype=torch.float32)
    l_all = torch.zeros((num_splits, b, h), dtype=torch.float32)
    o_all = torch.zeros((num_splits, b, h, d), dtype=torch.float32)

    for s in range(num_splits):
        start = s * split_size
        for bi in range(b):
            end = min(start + split_size, int(seq_lens[bi].item()))
            if start >= end:
                continue

            ks = kq[bi : bi + 1, :, start:end, :].float()
            vs = vq[bi : bi + 1, :, start:end, :].float()
            qb = q[bi : bi + 1].float()

            scores = torch.einsum("bhd,bhsd->bhs", qb, ks) * scale
            m = scores.amax(dim=-1)[0]
            e = torch.exp(scores - m[None, :, None])
            l = e.sum(dim=-1)[0]
            o = torch.einsum("bhs,bhsd->bhd", e, vs)[0]

            m_all[s, bi] = m
            l_all[s, bi] = l
            o_all[s, bi] = o

    return SplitState(m=m_all, l=l_all, o=o_all)


def merge_split_states(state: SplitState, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Merge (m,l,o) states while masking neutral/empty splits explicitly."""
    valid = state.l > 0
    safe_m = torch.where(valid, state.m, torch.full_like(state.m, -float("inf")))
    m = safe_m.amax(dim=0)

    # At least one split must be valid for every [B,H] because seq_len >= 1.
    if bool(torch.isneginf(m).any()):
        raise ValueError("cannot merge a sequence with no valid split")

    alpha = torch.where(valid, torch.exp(state.m - m.unsqueeze(0)), torch.zeros_like(state.m))
    denom = (alpha * state.l).sum(dim=0)
    numer = (alpha.unsqueeze(-1) * state.o).sum(dim=0)
    out = numer / denom.unsqueeze(-1)
    return out.to(out_dtype)


def split_kv_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    split_size: int,
    *,
    seq_lens: torch.Tensor | None = None,
    num_splits: int | None = None,
) -> torch.Tensor:
    state = split_kv_partials(
        q,
        k,
        v,
        split_size,
        seq_lens=seq_lens,
        num_splits=num_splits,
    )
    return merge_split_states(state, out_dtype=q.dtype)


def quantized_variant_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    variant: Variant | str,
    group_size: int = 32,
    split_size: int | None = None,
    seq_lens: torch.Tensor | None = None,
    num_splits: int | None = None,
) -> torch.Tensor:
    """Dense CPU reference for P/K/V/KV representation choices."""
    variant = variant if isinstance(variant, Variant) else Variant(variant)

    q_work = fwht_normalized(q) if variant.rotate_k else q
    k_store = fwht_normalized(k) if variant.rotate_k else k
    v_store = fwht_normalized(v) if variant.rotate_v else v

    k_qdq = qdq_int4_groupwise(k_store, group_size, out_dtype=q.dtype)
    v_qdq = qdq_int4_groupwise(v_store, group_size, out_dtype=q.dtype)

    if split_size is None:
        out = attention_decode(q_work, k_qdq, v_qdq, seq_lens=seq_lens)
    else:
        out = split_kv_attention(
            q_work,
            k_qdq,
            v_qdq,
            split_size,
            seq_lens=seq_lens,
            num_splits=num_splits,
        )

    if variant.rotate_v:
        out = fwht_normalized(out)
    return out


def paged_quantized_variant_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    variant: Variant | str,
    *,
    seq_lens: torch.Tensor,
    group_size: int = 32,
    page_size: int = 16,
    split_size: int | None = None,
    num_splits: int | None = None,
    seed: int = 1337,
) -> tuple[torch.Tensor, PagedInt4KV]:
    """CPU paged oracle exercising randomized physical-page placement."""
    variant = variant if isinstance(variant, Variant) else Variant(variant)
    q_work = fwht_normalized(q) if variant.rotate_k else q
    k_store = fwht_normalized(k) if variant.rotate_k else k
    v_store = fwht_normalized(v) if variant.rotate_v else v

    cache = pack_dense_to_paged_int4(
        k_store,
        v_store,
        seq_lens,
        page_size=page_size,
        group_size=group_size,
        seed=seed,
    )
    k_logical, v_logical = materialize_logical_cache(cache)

    if split_size is None:
        out = attention_decode(q_work, k_logical, v_logical, seq_lens=seq_lens)
    else:
        out = split_kv_attention(
            q_work,
            k_logical,
            v_logical,
            split_size,
            seq_lens=seq_lens,
            num_splits=num_splits,
        )
    if variant.rotate_v:
        out = fwht_normalized(out)
    return out, cache
