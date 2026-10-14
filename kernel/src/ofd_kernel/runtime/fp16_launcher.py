"""FP16 baseline with the frozen v2 split scheduling and shared merge."""
from ..triton.fp16_kernels import splitkv_fp16_partials_kernel
from ..triton.kernels import merge_splitkv_kernel
from .launcher import next_power_of_2


def fp16_paged_decode_core(q, k, v, bt, lens, *, config, workspace, max_seq_len):
    b,hq,d=q.shape
    pages,hkv,page,kd=k.shape
    assert kd==d and hq%hkv==0
    ns=workspace.partial_m.shape[2]
    assert ns==(max_seq_len+config.split_size-1)//config.split_size
    pm,pl,po,out=workspace.partial_m,workspace.partial_l,workspace.partial_o,workspace.out
    splitkv_fp16_partials_kernel[(b,hq,ns)](
        q,k,k,v,v,bt,lens,pm,pl,po,
        stride_q_b=q.stride(0),stride_q_h=q.stride(1),
        stride_k_page=k.stride(0),stride_k_head=k.stride(1),stride_k_tok=k.stride(2),stride_k_group=0,
        stride_ks_page=0,stride_ks_head=0,stride_ks_tok=0,stride_ks_group=0,
        stride_v_page=v.stride(0),stride_v_head=v.stride(1),stride_v_tok=v.stride(2),stride_v_group=0,
        stride_vs_page=0,stride_vs_head=0,stride_vs_tok=0,stride_vs_group=0,
        stride_bt_b=bt.stride(0),stride_bt_p=bt.stride(1),
        stride_pm_b=pm.stride(0),stride_pm_h=pm.stride(1),stride_pm_s=pm.stride(2),
        stride_po_b=po.stride(0),stride_po_h=po.stride(1),stride_po_s=po.stride(2),
        NUM_Q_HEADS=hq,NUM_KV_HEADS=hkv,HEAD_DIM=d,GROUP_SIZE=32,PAGE_SIZE=page,
        SPLIT_SIZE=config.split_size,TOKENS_PER_BLOCK=config.tokens_per_block,NUM_SPLITS=ns,
        num_warps=config.num_warps,num_stages=config.num_stages)
    merge_splitkv_kernel[(b,hq)](
        pm,pl,po,out,
        stride_pm_b=pm.stride(0),stride_pm_h=pm.stride(1),stride_pm_s=pm.stride(2),
        stride_po_b=po.stride(0),stride_po_h=po.stride(1),stride_po_s=po.stride(2),
        stride_o_b=out.stride(0),stride_o_h=out.stride(1),HEAD_DIM=d,NUM_SPLITS=ns,
        SPLIT_BLOCK=next_power_of_2(ns),num_warps=4,num_stages=1)
    return out
