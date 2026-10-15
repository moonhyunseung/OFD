"""OFD decode with preallocated GPU FWHT input/output buffers."""
from .launcher import triton_paged_decode_core

def gpu_ofd_decode(q,kp,ks,vp,vs,bt,lens,*,variant,config,workspace,max_seq_len,q_work,out_work):
    from ..triton.fwht import fwht_into
    query=q
    if variant.rotate_k:
        fwht_into(q,q_work)
        from ..triton.fwht import fwht_into
    query=q_work
    result=triton_paged_decode_core(query,kp,ks,vp,vs,bt,lens,config=config,workspace=workspace,max_seq_len=max_seq_len)
    if variant.rotate_v:
        fwht_into(result,out_work)
        return out_work
    return result
