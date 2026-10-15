from __future__ import annotations
import math,torch
from .launcher import (TritonDecodeConfig,TritonWorkspace,_validate_cuda_layout,allocate_workspace,
                       _validate_workspace,debug_validate_paged_metadata,next_power_of_2)
from ..triton.kernels import triton,merge_splitkv_kernel
from ..triton.grouped_kernels import grouped_int4_partials_kernel

def triton_paged_decode_grouped_core(
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

    grid1 = (b, hkv, ns)
    grouped_int4_partials_kernel[grid1](
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
