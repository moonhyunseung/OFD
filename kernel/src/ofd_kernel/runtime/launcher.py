from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from ..contracts import Variant
from ..reference.fwht import fwht_normalized
from ..triton.kernels import triton, splitkv_int4_partials_kernel, merge_splitkv_kernel


@dataclass
class TritonWorkspace:
    partial_m: torch.Tensor
    partial_l: torch.Tensor
    partial_o: torch.Tensor
    out: torch.Tensor
    max_seq_len: int


@dataclass(frozen=True)
class TritonDecodeConfig:
    split_size: int = 1024
    tokens_per_block: int = 16
    num_warps: int = 4
    num_stages: int = 2

    def __post_init__(self) -> None:
        if self.split_size <= 0 or self.tokens_per_block <= 0:
            raise ValueError("split_size and tokens_per_block must be positive")
        if self.split_size % self.tokens_per_block != 0:
            raise ValueError("split_size must be divisible by tokens_per_block")
        if self.tokens_per_block & (self.tokens_per_block - 1):
            raise ValueError("tokens_per_block must be a power of two")


def next_power_of_2(n: int) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    return 1 << (n - 1).bit_length()


def num_splits_for(seq_lens: torch.Tensor, split_size: int) -> int:
    if split_size <= 0:
        raise ValueError("split_size must be positive")
    if seq_lens.numel() == 0:
        raise ValueError("seq_lens cannot be empty")
    max_len = int(seq_lens.max().item())
    if max_len <= 0:
        raise ValueError("all seq_lens must be positive")
    return math.ceil(max_len / split_size)


def prepare_query(q: torch.Tensor, variant: Variant | str) -> torch.Tensor:
    variant = variant if isinstance(variant, Variant) else Variant(variant)
    return fwht_normalized(q) if variant.rotate_k else q


def finalize_output(out: torch.Tensor, variant: Variant | str) -> torch.Tensor:
    variant = variant if isinstance(variant, Variant) else Variant(variant)
    return fwht_normalized(out) if variant.rotate_v else out


def _validate_cuda_layout(
    q: torch.Tensor,
    k_packed: torch.Tensor,
    k_scale: torch.Tensor,
    v_packed: torch.Tensor,
    v_scale: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
) -> tuple[int, int, int, int, int, int]:
    tensors = (q, k_packed, k_scale, v_packed, v_scale, block_table, seq_lens)
    if not all(t.is_cuda for t in tensors):
        raise ValueError("all runtime tensors must be CUDA tensors")
    if q.ndim != 3:
        raise ValueError("q must be [B,Hq,D]")
    if k_packed.ndim != 5 or v_packed.ndim != 5:
        raise ValueError("packed K/V must be [P,Hkv,PAGE,G,BYTES_PER_GROUP]")
    if k_scale.ndim != 4 or v_scale.ndim != 4:
        raise ValueError("K/V scales must be [P,Hkv,PAGE,G]")
    if tuple(k_packed.shape) != tuple(v_packed.shape):
        raise ValueError("packed K/V shapes must match")
    if tuple(k_scale.shape) != tuple(v_scale.shape):
        raise ValueError("K/V scale shapes must match")

    b, hq, d = map(int, q.shape)
    _, hkv, page_size, groups, bytes_per_group = map(int, k_packed.shape)
    if hq % hkv != 0:
        raise ValueError("Hq must be divisible by Hkv")
    if block_table.ndim != 2 or block_table.shape[0] != b:
        raise ValueError("block_table must be [B,max_pages]")
    if seq_lens.shape != (b,):
        raise ValueError("seq_lens must be [B]")
    if block_table.dtype != torch.int32 or seq_lens.dtype != torch.int32:
        raise ValueError("block_table and seq_lens must be CUDA int32 tensors")
    if not block_table.is_contiguous() or not seq_lens.is_contiguous():
        raise ValueError("block_table and seq_lens must be contiguous")

    group_size = bytes_per_group * 2
    if groups * group_size != d:
        raise ValueError("packed layout does not match q head_dim")
    if d & (d - 1):
        raise ValueError("head_dim must be a power of two")
    if not q.is_contiguous() or not k_packed.is_contiguous() or not v_packed.is_contiguous():
        raise ValueError("baseline launcher requires contiguous q/packed K/V")
    return b, hq, hkv, d, page_size, group_size



def allocate_workspace(
    q: torch.Tensor,
    seq_lens: torch.Tensor | None,
    config: TritonDecodeConfig,
    *,
    max_seq_len: int | None = None,
) -> TritonWorkspace:
    """Allocate split-KV workspace outside the timed decode path.

    Production callers should pass ``max_seq_len`` from host-side scheduler
    metadata. Falling back to ``seq_lens.max().item()`` is intentionally
    allowed only during setup/debug because it synchronizes CUDA.
    """
    if not q.is_cuda:
        raise ValueError("workspace is for CUDA execution only")
    b, hq, d = map(int, q.shape)
    if max_seq_len is None:
        if seq_lens is None:
            raise ValueError("provide max_seq_len or seq_lens")
        max_seq_len = int(seq_lens.max().item())
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    ns = math.ceil(max_seq_len / config.split_size)
    partial_m = torch.empty((b, hq, ns), device=q.device, dtype=torch.float32)
    partial_l = torch.empty_like(partial_m)
    partial_o = torch.empty((b, hq, ns, d), device=q.device, dtype=torch.float32)
    out = torch.empty((b, hq, d), device=q.device, dtype=q.dtype)
    return TritonWorkspace(partial_m, partial_l, partial_o, out, int(max_seq_len))


def _validate_workspace(
    workspace: TritonWorkspace,
    q: torch.Tensor,
    ns: int,
) -> None:
    b, hq, d = map(int, q.shape)
    expected_m = (b, hq, ns)
    expected_o = (b, hq, ns, d)
    if tuple(workspace.partial_m.shape) != expected_m:
        raise ValueError(f"workspace.partial_m shape must be {expected_m}")
    if tuple(workspace.partial_l.shape) != expected_m:
        raise ValueError(f"workspace.partial_l shape must be {expected_m}")
    if tuple(workspace.partial_o.shape) != expected_o:
        raise ValueError(f"workspace.partial_o shape must be {expected_o}")
    if tuple(workspace.out.shape) != (b, hq, d):
        raise ValueError("workspace.out shape mismatch")
    if workspace.out.device != q.device:
        raise ValueError("workspace device mismatch")

def debug_validate_paged_metadata(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    page_size: int,
    num_physical_pages: int,
    max_seq_len: int,
) -> None:
    """Expensive synchronization-based validation for correctness/debug runs.

    Copies metadata to CPU and validates only logical pages actually used by
    each request. Never enable this inside a performance timing loop.
    """
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    lens_cpu = seq_lens.detach().to(device="cpu", dtype=torch.int64)
    bt_cpu = block_table.detach().to(device="cpu", dtype=torch.int64)
    if lens_cpu.numel() == 0 or int(lens_cpu.min().item()) <= 0:
        raise ValueError("seq_lens must be positive")
    actual_max = int(lens_cpu.max().item())
    if actual_max > max_seq_len:
        raise ValueError(f"seq_lens max {actual_max} exceeds declared max_seq_len {max_seq_len}")
    max_needed_pages = math.ceil(actual_max / page_size)
    if block_table.shape[1] < max_needed_pages:
        raise ValueError(
            f"block_table has {block_table.shape[1]} logical pages but {max_needed_pages} are required"
        )
    for b in range(int(lens_cpu.numel())):
        needed = math.ceil(int(lens_cpu[b].item()) / page_size)
        used = bt_cpu[b, :needed]
        if used.numel() == 0:
            continue
        lo = int(used.min().item())
        hi = int(used.max().item())
        if lo < 0 or hi >= num_physical_pages:
            raise ValueError(
                f"request {b} block_table physical page out of range: "
                f"min={lo}, max={hi}, num_physical_pages={num_physical_pages}"
            )


def triton_paged_decode_core(
    q_work: torch.Tensor,
    k_packed: torch.Tensor,
    k_scale: torch.Tensor,
    v_packed: torch.Tensor,
    v_scale: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    config: TritonDecodeConfig = TritonDecodeConfig(),
    workspace: TritonWorkspace | None = None,
    max_seq_len: int | None = None,
    debug_validate: bool = False,
) -> torch.Tensor:
    """Branch-free packed-INT4 attention core with no Q/output FWHT.

    This is the function used for core-kernel timing. A provided workspace is
    required for synchronization-free timed execution.
    """
    if triton is None:
        raise RuntimeError("Triton is not importable in this environment")
    b, hq, hkv, d, page_size, group_size = _validate_cuda_layout(
        q_work, k_packed, k_scale, v_packed, v_scale, block_table, seq_lens
    )

    if workspace is None:
        workspace = allocate_workspace(
            q_work, seq_lens, config, max_seq_len=max_seq_len
        )
    ns = int(workspace.partial_m.shape[2])
    if max_seq_len is None:
        max_seq_len = int(workspace.max_seq_len)
    expected_ns = math.ceil(max_seq_len / config.split_size)
    if ns != expected_ns:
        raise ValueError(f"workspace NUM_SPLITS={ns} but declared max_seq_len requires {expected_ns}")
    _validate_workspace(workspace, q_work, ns)

    if debug_validate:
        debug_validate_paged_metadata(
            block_table, seq_lens, page_size=page_size,
            num_physical_pages=int(k_packed.shape[0]), max_seq_len=max_seq_len,
        )

    split_block = next_power_of_2(ns)
    partial_m, partial_l, partial_o, out = (
        workspace.partial_m, workspace.partial_l, workspace.partial_o, workspace.out
    )

    # The hot path accepts only pre-normalized metadata; no hidden dtype/layout
    # conversion or CUDA->CPU synchronization occurs here.
    bt = block_table
    lens = seq_lens

    grid1 = (b, hq, ns)
    splitkv_int4_partials_kernel[grid1](
        q_work, k_packed, k_scale, v_packed, v_scale, bt, lens,
        partial_m, partial_l, partial_o,
        stride_q_b=q_work.stride(0), stride_q_h=q_work.stride(1),
        stride_k_page=k_packed.stride(0), stride_k_head=k_packed.stride(1),
        stride_k_tok=k_packed.stride(2), stride_k_group=k_packed.stride(3),
        stride_ks_page=k_scale.stride(0), stride_ks_head=k_scale.stride(1),
        stride_ks_tok=k_scale.stride(2), stride_ks_group=k_scale.stride(3),
        stride_v_page=v_packed.stride(0), stride_v_head=v_packed.stride(1),
        stride_v_tok=v_packed.stride(2), stride_v_group=v_packed.stride(3),
        stride_vs_page=v_scale.stride(0), stride_vs_head=v_scale.stride(1),
        stride_vs_tok=v_scale.stride(2), stride_vs_group=v_scale.stride(3),
        stride_bt_b=bt.stride(0), stride_bt_p=bt.stride(1),
        stride_pm_b=partial_m.stride(0), stride_pm_h=partial_m.stride(1),
        stride_pm_s=partial_m.stride(2),
        stride_po_b=partial_o.stride(0), stride_po_h=partial_o.stride(1),
        stride_po_s=partial_o.stride(2),
        NUM_Q_HEADS=hq, NUM_KV_HEADS=hkv, HEAD_DIM=d, GROUP_SIZE=group_size,
        PAGE_SIZE=page_size, SPLIT_SIZE=config.split_size,
        TOKENS_PER_BLOCK=config.tokens_per_block, NUM_SPLITS=ns,
        num_warps=config.num_warps, num_stages=config.num_stages,
    )

    grid2 = (b, hq)
    merge_splitkv_kernel[grid2](
        partial_m, partial_l, partial_o, out,
        stride_pm_b=partial_m.stride(0), stride_pm_h=partial_m.stride(1),
        stride_pm_s=partial_m.stride(2),
        stride_po_b=partial_o.stride(0), stride_po_h=partial_o.stride(1),
        stride_po_s=partial_o.stride(2),
        stride_o_b=out.stride(0), stride_o_h=out.stride(1),
        HEAD_DIM=d, NUM_SPLITS=ns, SPLIT_BLOCK=split_block,
        num_warps=4, num_stages=1,
    )
    return out


def triton_paged_decode(
    q: torch.Tensor,
    k_packed: torch.Tensor,
    k_scale: torch.Tensor,
    v_packed: torch.Tensor,
    v_scale: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    variant: Variant | str = Variant.P,
    config: TritonDecodeConfig = TritonDecodeConfig(),
    workspace: TritonWorkspace | None = None,
    max_seq_len: int | None = None,
    debug_validate: bool = False,
) -> torch.Tensor:
    """End-to-end reference wrapper around the Triton attention core.

    Q/output transforms still use the PyTorch FWHT reference. Therefore this
    wrapper is valid for correctness but is *not* claim-eligible for OFD
    end-to-end latency until a production GPU FWHT path exists.
    """
    variant = variant if isinstance(variant, Variant) else Variant(variant)
    q_work = prepare_query(q, variant).contiguous()
    out = triton_paged_decode_core(
        q_work, k_packed, k_scale, v_packed, v_scale, block_table, seq_lens,
        config=config, workspace=workspace, max_seq_len=max_seq_len,
        debug_validate=debug_validate,
    )
    return finalize_output(out, variant)
