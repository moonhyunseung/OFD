"""Preallocated online paged KV, optional independent K/V rotation per layer."""
import math,types,torch
from ..contracts import Variant
from ..reference.fwht import fwht_normalized
from ..reference.quant import quantize_int4_packed_groupwise
from ..triton.fwht import fwht_into
from ..triton.append import append_token
from .launcher import TritonDecodeConfig,allocate_workspace
from .gpu_ofd import gpu_ofd_decode
from .fp16_launcher import fp16_paged_decode_core
from .subgrouped_launcher import subgrouped_paged_decode_core

CFG=TritonDecodeConfig()

def move_cache(cache,device):
    """Cache templates live on CPU; all transfers occur before decode timing."""
    for layer in cache.layers:
        for key,val in vars(layer).items():
            if isinstance(val,torch.Tensor):setattr(layer,key,val.to(device))
    return cache

class OnlineKV:
    def __init__(self,k,v,variant,steps,seed=1337,group_q=1):
        if group_q not in (1,2,3,6):raise ValueError("group_q must be 1,2,3,6")
        if group_q != 1 and variant not in ("P","FP16"):raise ValueError("E10 grouping is P/FP16 only")
        self.group_q=group_q
        self.variant=variant;self.start=k.shape[2];self.pos=self.start
        b,h,t,d=k.shape;self.capacity=t+steps;np=math.ceil(self.capacity/16)
        g=torch.Generator(device=k.device).manual_seed(seed)
        perm=torch.randperm(b*np,generator=g,device=k.device)
        self.bt=perm.reshape(b,np).to(torch.int32)
        def page(x):
            buf=torch.zeros((b,h,np*16,d),device=x.device,dtype=x.dtype);buf[:,:,:t].copy_(x)
            logical=buf.reshape(b,h,np,16,d).permute(0,2,1,3,4).reshape(b*np,h,16,d).contiguous()
            physical=torch.empty_like(logical);physical[perm]=logical
            return physical
        kk=page(k);vv=page(v)
        if variant=='FP16':self.kp=kk;self.vp=vv;self.ks=kk;self.vs=vv
        else:
            vk=Variant(variant)
            if vk.rotate_k:kk=fwht_normalized(kk)
            if vk.rotate_v:vv=fwht_normalized(vv)
            def pack(x):
                o=quantize_int4_packed_groupwise(x,32)
                return o.packed.reshape(b*np,h,16,d//32,16),o.scale.reshape(b*np,h,16,d//32)
            self.kp,self.ks=pack(kk);self.vp,self.vs=pack(vv)
        self.q=torch.empty((b,24,d),device=k.device,dtype=k.dtype)
        self.qw=torch.empty_like(self.q);self.ow=torch.empty_like(self.q)
        self.kin=torch.empty((b,h,d),device=k.device,dtype=k.dtype);self.vin=torch.empty_like(self.kin)
        self.kr=torch.empty_like(self.kin);self.vr=torch.empty_like(self.kin)
        self.lens=[torch.full((b,),t+i+1,device=k.device,dtype=torch.int32) for i in range(steps)]
        self.ws=allocate_workspace(self.q,None,CFG,max_seq_len=self.capacity)
    def append(self,k,v):
        self.kin.copy_(k[:,:,0,:]);self.vin.copy_(v[:,:,0,:]);kk=self.kin;vv=self.vin
        if self.variant!='FP16':
            vr=Variant(self.variant)
            if vr.rotate_k:fwht_into(kk,self.kr);kk=self.kr
            if vr.rotate_v:fwht_into(vv,self.vr);vv=self.vr
        append_token(kk,vv,self.kp,self.ks,self.vp,self.vs,self.bt,self.pos)
        self.pos+=1
    def attention(self,q):
        self.q.copy_(q[:,:,0,:]);lens=self.lens[self.pos-self.start-1]
        if self.group_q != 1:
            out=subgrouped_paged_decode_core(self.q,self.kp,self.ks,self.vp,self.vs,self.bt,lens,config=CFG,workspace=self.ws,max_seq_len=self.capacity,representation='FP16' if self.variant=='FP16' else 'INT4',group_q=self.group_q)
        elif self.variant=='FP16':
            out=fp16_paged_decode_core(self.q,self.kp,self.vp,self.bt,lens,config=CFG,workspace=self.ws,max_seq_len=self.capacity)
        else:
            out=gpu_ofd_decode(self.q,self.kp,self.ks,self.vp,self.vs,self.bt,lens,variant=Variant(self.variant),config=CFG,workspace=self.ws,max_seq_len=self.capacity,q_work=self.qw,out_work=self.ow)
        return out[:,None,:,:]
    @property
    def kv_bytes(self):
        xs=[self.kp,self.vp] if self.variant=='FP16' else [self.kp,self.ks,self.vp,self.vs]
        return sum(x.numel()*x.element_size() for x in xs)

class NativeKV:
    """Native attention with preallocated dense KV, excluding cache reallocation."""
    def __init__(self,k,v,steps):
        b,h,t,d=k.shape;self.pos=t
        self.k=torch.empty((b,h,t+steps,d),device=k.device,dtype=k.dtype);self.v=torch.empty_like(self.k)
        self.k[:,:,:t].copy_(k);self.v[:,:,:t].copy_(v)
    def append(self,k,v):
        self.k[:,:,self.pos:self.pos+1].copy_(k);self.v[:,:,self.pos:self.pos+1].copy_(v);self.pos+=1
    @property
    def kv_bytes(self):return (self.k.numel()+self.v.numel())*self.k.element_size()

def install(cache,policy,steps,group_q=1):
    stores={}
    for index,variant in policy.items():
        layer=cache.layers[int(index)];store=NativeKV(layer.keys,layer.values,steps) if variant=='NATIVE' else OnlineKV(layer.keys,layer.values,variant,steps,group_q=group_q)
        stores[int(index)]=store;layer.keys=None;layer.values=None
        def update(self,k,v,*args,_s=store,**kwargs):
            _s.append(k,v)
            if isinstance(_s,NativeKV):return _s.k[:,:,:_s.pos,:],_s.v[:,:,:_s.pos,:]
            return k,v
        def length(self,*args,_s=store,**kwargs):return _s.pos
        layer.update=types.MethodType(update,layer);layer.get_seq_length=types.MethodType(length,layer)
    return stores
