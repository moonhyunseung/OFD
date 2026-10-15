"""Online single-token paged KV writes. Buffers and metadata preallocated."""
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice

@triton.jit(do_not_specialize=["POS"])
def _append(K,V,KP,KS,VP,VS,BT,POS,NP:tl.constexpr,H:tl.constexpr,D:tl.constexpr,INT4:tl.constexpr):
    b=tl.program_id(0);h=tl.program_id(1)
    d=tl.arange(0,D)
    physical=tl.load(BT+b*NP+POS//16)
    base=((physical*H+h)*16+POS%16)
    k=tl.load(K+(b*H+h)*D+d).to(tl.float32)
    v=tl.load(V+(b*H+h)*D+d).to(tl.float32)
    if INT4:
        kg=tl.reshape(k,(D//32,32));vg=tl.reshape(v,(D//32,32))
        sk=tl.maximum(tl.max(tl.abs(kg),1),1.0e-8)/7.0
        sv=tl.maximum(tl.max(tl.abs(vg),1),1.0e-8)/7.0
        ck=(libdevice.nearbyint(kg/sk[:,None]).to(tl.int32)+8).to(tl.uint8)
        cv=(libdevice.nearbyint(vg/sv[:,None]).to(tl.int32)+8).to(tl.uint8)
        pair=tl.arange(0,16)
        ix=tl.arange(0,D//32)[:,None]*0+pair[None,:]*2
        klo=tl.gather(ck,ix,1);khi=tl.gather(ck,ix+1,1)
        vlo=tl.gather(cv,ix,1);vhi=tl.gather(cv,ix+1,1)
        offsets=base*(D//2)+tl.arange(0,D//32)[:,None]*16+pair[None,:]
        tl.store(KP+offsets,klo|(khi<<4));tl.store(VP+offsets,vlo|(vhi<<4))
        tl.store(KS+base*(D//32)+tl.arange(0,D//32),sk)
        tl.store(VS+base*(D//32)+tl.arange(0,D//32),sv)
    else:
        tl.store(KP+base*D+d,k);tl.store(VP+base*D+d,v)

def append_token(k,v,kp,ks,vp,vs,bt,pos):
    b,h,d=k.shape
    _append[(b,h)](k,v,kp,ks,vp,vs,bt,POS=pos,NP=bt.shape[1],H=h,D=d,INT4=kp.dtype.__str__()=='torch.uint8',num_warps=4)
