"""E05 core-only matched paged FP16/INT4 KV-budget experiment."""
import argparse,csv,gc,hashlib,json,math,subprocess
from pathlib import Path
import torch,triton
from bench_gpu import cuda_ms
from ofd_kernel.reference.quant import quantize_int4_packed_groupwise
from ofd_kernel.runtime.launcher import TritonDecodeConfig,allocate_workspace,triton_paged_decode_core
from ofd_kernel.runtime.fp16_launcher import fp16_paged_decode_core

CFG=TritonDecodeConfig()
H=8; D=128; P=16; G=32


def correctness():
    torch.manual_seed(1337)
    b,t=3,257
    n=math.ceil(t/P)
    k=torch.randn(b*n,H,P,D,device='cuda',dtype=torch.float16)
    v=torch.randn_like(k)
    bt=torch.randperm(b*n,device='cuda').reshape(b,n).int()
    lens=torch.tensor([17,93,257],device='cuda',dtype=torch.int32)
    q=torch.randn(b,H,D,device='cuda',dtype=torch.float16)
    ws=allocate_workspace(q,lens,CFG,max_seq_len=t)
    got=fp16_paged_decode_core(q,k,v,bt,lens,config=CFG,workspace=ws,max_seq_len=t).clone()
    ref=[]
    for i,L in enumerate([17,93,257]):
        kk=k[bt[i].long()].permute(1,0,2,3).reshape(H,-1,D)[:,:L].float()
        vv=v[bt[i].long()].permute(1,0,2,3).reshape(H,-1,D)[:,:L].float()
        a=torch.softmax(torch.sum(kk*q[i].float()[:,None,:],-1)/math.sqrt(D),-1)
        ref.append(torch.sum(a[:,:,None]*vv,1))
    ref=torch.stack(ref)
    error=((got.float()-ref).norm()/(ref.norm()+1e-12)).item()
    print('FP16_PAGED_CORRECTNESS',error,flush=True)
    if not math.isfinite(error) or error>0.002: raise RuntimeError('FP16 numerical gate failed')
    return error


def pair(t,bf,seed,warmup,repeat):
    bi=(bf*32)//9
    n=t//P
    torch.manual_seed(seed)
    q=torch.randn(bi,H,D,device='cuda',dtype=torch.float16)
    # Disjoint randomized pages per request; FP16 uses the identical first bf rows.
    bt=torch.stack([torch.randperm(n,device='cuda')+i*n for i in range(bi)]).int().contiguous()
    lens=torch.full((bi,),t,device='cuda',dtype=torch.int32)
    kp=torch.empty(bi*n,H,P,D//G,G//2,device='cuda',dtype=torch.uint8)
    vp=torch.empty_like(kp)
    ks=torch.empty(bi*n,H,P,D//G,device='cuda',dtype=torch.float16)
    vs=torch.empty_like(ks)
    k=torch.empty(bf*n,H,P,D,device='cuda',dtype=torch.float16)
    v=torch.empty_like(k)
    # Common original FP16 values for shared requests; exact absmax/7 quantizer.
    for start in range(0,bi*n,128):
        end=min(start+128,bi*n)
        for packed,scales,fp in [(kp,ks,k),(vp,vs,v)]:
            x=torch.randn(end-start,H,P,D,device='cuda',dtype=torch.float16)
            obj=quantize_int4_packed_groupwise(x,G)
            packed[start:end].copy_(obj.packed.reshape(end-start,H,P,D//G,G//2))
            scales[start:end].copy_(obj.scale.reshape(end-start,H,P,D//G))
            if start<bf*n:
                stop=min(end,bf*n)
                fp[start:stop].copy_(x[:stop-start])
            del obj,x
    qf=q[:bf].contiguous(); btf=bt[:bf].contiguous(); lf=lens[:bf].contiguous()
    wf=allocate_workspace(qf,lf,CFG,max_seq_len=t)
    wi=allocate_workspace(q,lens,CFG,max_seq_len=t)
    fp_bytes=sum(x.numel()*x.element_size() for x in (k,v))
    int_bytes=sum(x.numel()*x.element_size() for x in (kp,ks,vp,vs))
    if int_bytes>fp_bytes: raise RuntimeError('INT4 exceeds matched KV budget')
    def fp(): return fp16_paged_decode_core(qf,k,v,btf,lf,config=CFG,workspace=wf,max_seq_len=t)
    def iq(): return triton_paged_decode_core(q,kp,ks,vp,vs,bt,lens,config=CFG,workspace=wi,max_seq_len=t)
    fp();iq();torch.cuda.synchronize()
    if not torch.isfinite(wf.out).all().item() or not torch.isfinite(wi.out).all().item():
        raise RuntimeError('Nonfinite sweep output')
    fm=cuda_ms(fp,warmup,repeat); im=cuda_ms(iq,warmup,repeat)
    return dict(fp16_batch=bf,int4_batch=bi,fp16_kv_bytes=fp_bytes,int4_kv_bytes=int_bytes,
                int4_fp16_memory_ratio=int_bytes/fp_bytes,resident_batch_ratio=bi/bf,
                fp16_ms=fm,int4_ms=im,fp16_tok_s=bf*1000/fm,int4_tok_s=bi*1000/im,
                int4_fp16_throughput_gain=(bi/im)/(bf/fm),status='OK',
                fp16_workspace_bytes=sum(x.numel()*x.element_size() for x in (wf.partial_m,wf.partial_l,wf.partial_o,wf.out)),
                int4_workspace_bytes=sum(x.numel()*x.element_size() for x in (wi.partial_m,wi.partial_l,wi.partial_o,wi.out)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    error=correctness()
    contexts=[8192] if a.smoke else [8192,16384,32768,65536,131072]
    batches=[1] if a.smoke else [1,2,4,8]
    sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    if dirty: raise RuntimeError('Dirty worktree')
    driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip()
    common=dict(gpu=torch.cuda.get_device_name(0),driver=driver,torch=torch.__version__,cuda=torch.version.cuda,triton=triton.__version__,git_sha=sha,
                hkv=H,q_per_kv=1,head_dim=D,page_size=P,group_size=G,dtype='fp16',
                kernel_config=json.dumps(CFG.__dict__,sort_keys=True),timing_scope='core_attention_only',fp16_correctness_rel_error=error)
    fields=list(common)+['context','seed','fp16_batch','int4_batch','fp16_kv_bytes','int4_kv_bytes','int4_fp16_memory_ratio','resident_batch_ratio',
                        'fp16_ms','int4_ms','fp16_tok_s','int4_tok_s','int4_fp16_throughput_gain','fp16_workspace_bytes','int4_workspace_bytes','status']
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for t in contexts:
            for b in batches:
                seed=1337+t+b
                try: row=pair(t,b,seed,3,10)
                except torch.OutOfMemoryError: row=dict(fp16_batch=b,int4_batch=(b*32)//9,status='OOM')
                row.update(common,context=t,seed=seed)
                w.writerow(row);f.flush()
                print(t,b,row.get('int4_fp16_throughput_gain'),row['status'],flush=True)
                gc.collect();torch.cuda.empty_cache()

if __name__=='__main__': main()
