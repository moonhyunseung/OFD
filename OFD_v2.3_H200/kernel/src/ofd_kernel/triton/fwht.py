"""Allocation-free normalized GPU FWHT, one power-of-two row per program."""
import triton
import triton.language as tl

@triton.jit
def _fwht(X,Y,D:tl.constexpr,LOG_D:tl.constexpr):
    row=tl.program_id(0)
    d=tl.arange(0,D)
    x=tl.load(X+row*D+d).to(tl.float32)
    for stage in tl.static_range(LOG_D):
        step=1<<stage
        partner=tl.gather(x,d^step,axis=0)
        x=tl.where((d&step)==0,x+partner,partner-x)
    tl.store(Y+row*D+d,x*(1.0/tl.sqrt(float(D))))

def fwht_into(x,out):
    if not x.is_cuda or not out.is_cuda or x.shape!=out.shape or x.dtype!=out.dtype:
        raise ValueError('Matching CUDA input/output required')
    d=x.shape[-1]
    if d<1 or d&(d-1) or d>256 or not x.is_contiguous() or not out.is_contiguous():
        raise ValueError('Contiguous power-of-two rows up to 256 required')
    _fwht[(x.numel()//d,)](x,out,D=d,LOG_D=d.bit_length()-1,num_warps=4)
    return out
