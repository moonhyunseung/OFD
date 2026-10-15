"""Exact append storage check and quantized attention oracle, no timing claims."""
import torch
import torch.nn.functional as F
from ofd_kernel.runtime.online_cache import OnlineKV
from ofd_kernel.reference.fwht import fwht_normalized
from ofd_kernel.reference.quant import quantize_int4_packed_groupwise
from bench_gpu import rel

def check():
    torch.manual_seed(1337);b,h,t,d=2,4,17,256
    k=torch.randn(b,h,t,d,device='cuda',dtype=torch.float16);v=torch.randn_like(k)
    nk=torch.randn(b,h,3,d,device='cuda',dtype=torch.float16);nv=torch.randn_like(nk);q=torch.randn(b,24,1,d,device='cuda',dtype=torch.float16)
    rows=[]
    for variant in ['FP16','P','K','V','KV']:
        store=OnlineKV(k,v,variant,3)
        for s in range(3):store.append(nk[:,:,s:s+1],nv[:,:,s:s+1])
        kk=torch.cat([k,nk],2);vv=torch.cat([v,nv],2)
        if variant in ['K','KV']:kk=fwht_normalized(kk)
        if variant in ['V','KV']:vv=fwht_normalized(vv)
        def logical(x):return x[store.bt.long()].permute(0,2,1,3,4).reshape(b,h,-1,x.shape[-1])[:,:,:t+3]
        if variant=='FP16':
            ak=logical(store.kp);av=logical(store.vp)
            assert torch.equal(ak,kk) and torch.equal(av,vv)
        else:
            def unpack_check(x,packed,scales):
                expected=quantize_int4_packed_groupwise(x,32)
                ap=packed[store.bt.long()].permute(0,2,1,3,4,5).reshape(b,h,-1,d//32,16)[:,:,:t+3]
                ass=scales[store.bt.long()].permute(0,2,1,3,4).reshape(b,h,-1,d//32)[:,:,:t+3]
                assert torch.equal(ap,expected.packed.reshape_as(ap)),f'{variant} packed codes mismatch'
                assert torch.equal(ass,expected.scale.reshape_as(ass)),f'{variant} scales mismatch'
                lo=ap&15;hi=ap>>4;codes=torch.stack([lo,hi],-1).reshape(b,h,t+3,d//32,32).float()-8
                return (codes*ass[...,None]).reshape(b,h,t+3,d).half()
            ak=unpack_check(kk,store.kp,store.ks);av=unpack_check(vv,store.vp,store.vs)
        qq=fwht_normalized(q) if variant in ['K','KV'] else q
        oracle=F.scaled_dot_product_attention(qq,ak,av,enable_gqa=True).transpose(1,2)
        if variant in ['V','KV']:oracle=fwht_normalized(oracle)
        actual=store.attention(q);error=rel(actual,oracle)
        if error>0.002:raise RuntimeError(f'Online oracle error {variant} {error}')
        rows.append(dict(variant=variant,relative_error=error,storage_exact=True))
    print('ONLINE_KV_VALIDATION',rows,flush=True)
    return rows
if __name__=='__main__':check()
