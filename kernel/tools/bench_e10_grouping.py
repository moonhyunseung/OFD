"""E10 only: reuse frozen kernels, quantizer and cuda_ms; no E00-E09 changes."""
import argparse,csv,gc,json,math,os,random,subprocess,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
import torch,triton
from bench_gpu import cuda_ms
from ofd_kernel.reference.quant import quantize_int4_packed_groupwise
from ofd_kernel.runtime.launcher import TritonDecodeConfig,allocate_workspace,triton_paged_decode_core
from ofd_kernel.runtime.fp16_launcher import fp16_paged_decode_core
from ofd_kernel.runtime.grouped_launcher import triton_paged_decode_grouped_core
from ofd_kernel.runtime.subgrouped_launcher import subgrouped_paged_decode_core
ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'kernel/experiments/E10_gqa_grouping/protocol.json'
CFG=TritonDecodeConfig(split_size=1024,tokens_per_block=16,num_warps=4,num_stages=2)

def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def error(ref,out):
    return ((out.float()-ref.float()).norm()/(ref.float().norm()+1e-12)).item()

def cache(t,b,seed):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    n=b*(t//16);shape=(n,4,16,256)
    k=torch.empty(shape,device='cuda',dtype=torch.float16);v=torch.empty_like(k)
    kp=torch.empty(n,4,16,8,16,device='cuda',dtype=torch.uint8);vp=torch.empty_like(kp)
    ks=torch.empty(n,4,16,8,device='cuda',dtype=torch.float16);vs=torch.empty_like(ks)
    for start in range(0,n,128):
        end=min(start+128,n)
        for x,pk,sc in [(k,kp,ks),(v,vp,vs)]:
            z=torch.randn(end-start,4,16,256,device='cuda',dtype=torch.float16);x[start:end].copy_(z)
            qz=quantize_int4_packed_groupwise(z,32)
            pk[start:end].copy_(qz.packed.reshape(end-start,4,16,8,16));sc[start:end].copy_(qz.scale.reshape(end-start,4,16,8))
    bt=torch.randperm(n,device='cuda').reshape(b,t//16).int().contiguous()
    lens=torch.full((b,),t,device='cuda',dtype=torch.int32)
    q=torch.randn(b,24,256,device='cuda',dtype=torch.float16)
    return q,k,v,kp,ks,vp,vs,bt,lens

def oracle(q,k,v,kp,ks,vp,vs,bt,t,representation):
    # Stable chunked FP32 softmax, no timing. Same actual quantized KV for INT4.
    outs=[]
    for bi in range(q.shape[0]):
        page_ids=bt[bi].long();out=[]
        for h in range(24):
            kvh=h//6;m=torch.tensor(-float('inf'),device='cuda');l=torch.zeros((),device='cuda');acc=torch.zeros(256,device='cuda')
            for start in range(0,page_ids.numel(),64):
                ids=page_ids[start:start+64]
                def values(x,pk,sc):
                    if representation=='FP16':return x[ids,kvh].reshape(-1,256).float()
                    packed=pk[ids,kvh].int();codes=torch.stack((packed&15,(packed>>4)&15),dim=-1).reshape(-1,16,8,32)-8
                    return (codes.float()*sc[ids,kvh].float()[...,None]).reshape(-1,256)
                kk=values(k,kp,ks);vv=values(v,vp,vs)
                score=(kk*q[bi,h].float()).sum(-1)/16;mnew=torch.maximum(m,score.max());alpha=(m-mnew).exp();p=(score-mnew).exp()
                acc=acc*alpha+(p[:,None]*vv).sum(0);l=l*alpha+p.sum();m=mnew
            out.append(acc/l)
        outs.append(torch.stack(out))
    return torch.stack(outs)

def worker(out,process,smoke):
    protocol=json.loads(PROTOCOL.read_text());sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    env=dict(git_sha=sha,GPU=torch.cuda.get_device_name(),CUDA=torch.version.cuda,PyTorch=torch.__version__,Triton=triton.__version__)
    raw=[];checks=[];fail=[]
    shapes=[(2048,1)] if smoke else [(t,b) for t in protocol['contexts'] for b in protocol['batches']]
    random.Random(1337+process).shuffle(shapes)
    for t,b in shapes:
        seed=1337+t+b;print('SHAPE',process,t,b,flush=True)
        try:
            q,k,v,kp,ks,vp,vs,bt,lens=cache(t,b,seed)
            ws=allocate_workspace(q,lens,CFG,max_seq_len=t)
            def legacy(rep):
                if rep=='INT4':return triton_paged_decode_core(q,kp,ks,vp,vs,bt,lens,config=CFG,workspace=ws,max_seq_len=t)
                return fp16_paged_decode_core(q,k,v,bt,lens,config=CFG,workspace=ws,max_seq_len=t)
            def generalized(rep,g):
                return subgrouped_paged_decode_core(q,kp if rep=='INT4' else k,ks if rep=='INT4' else k,vp if rep=='INT4' else v,vs if rep=='INT4' else v,bt,lens,config=CFG,workspace=ws,max_seq_len=t,representation=rep,group_q=g)
            refs={rep:legacy(rep).clone() for rep in ['INT4','FP16']}
            errors={}
            for rep in ['INT4','FP16']:
                if smoke:
                    ref=oracle(q,k,v,kp,ks,vp,vs,bt,t,rep);err=error(ref,refs[rep]);finite=bool(torch.isfinite(refs[rep]).all().item())
                    checks.append(dict(process_id=process,context=t,batch=b,representation=rep,group_q=1,reference='FP32_oracle',relative_error=err,finite=finite,git_sha=sha));write_csv(out/f'correctness_p{process}.csv',checks)
                    if not finite or err>.002:raise RuntimeError('Legacy oracle mismatch')
                for g in [1,2,3,6]:
                    val=generalized(rep,g);err=error(refs[rep],val);finite=bool(torch.isfinite(val).all().item());errors[rep,g]=err
                    checks.append(dict(process_id=process,context=t,batch=b,representation=rep,group_q=g,reference='legacy_g1',relative_error=err,finite=finite,git_sha=sha));write_csv(out/f'correctness_p{process}.csv',checks)
                    if not finite or err>.002:raise RuntimeError(f'Correctness {rep} g{g}: {err}')
                val=generalized(rep,6).clone()
                if rep=='INT4':
                    old=triton_paged_decode_grouped_core(q,kp,ks,vp,vs,bt,lens,config=CFG,workspace=ws,max_seq_len=t)
                    err=error(old,val);finite=bool(torch.isfinite(old).all().item())
                    checks.append(dict(process_id=process,context=t,batch=b,representation=rep,group_q=6,reference='existing_grouped_padded',relative_error=err,finite=finite,git_sha=sha));write_csv(out/f'correctness_p{process}.csv',checks)
                    if not finite or err>.002:raise RuntimeError('Grouped g6 mismatch')
            variants=[(rep,g,'generalized') for rep in ['INT4','FP16'] for g in [1,2,3,6]]+[(rep,1,'legacy') for rep in ['INT4','FP16']]
            # All variants compiled/validated above; repeated warmup before every timed block.
            for round_id in range(protocol['rounds']):
                order=variants.copy();random.Random(1337+process*100003+t+b*17+round_id).shuffle(order)
                for ordinal,(rep,g,path) in enumerate(order):
                    fn=(lambda rep=rep:legacy(rep)) if path=='legacy' else (lambda rep=rep,g=g:generalized(rep,g))
                    ms=cuda_ms(fn,protocol['warmup'],protocol['repeat']);ns=(t+1023)//1024
                    row=dict(context=t,batch=b,representation=rep,group_q=g,path=path,Hq=24,Hkv=4,head_dim=256,num_splits=ns,stage1_program_count=b*4*(6//g)*ns,W1=b*24*ns,latency_ms=ms,aggregate_tok_s=b*1000/ms,relative_error=errors[rep,g] if path!='legacy' else 0.0,split_size=1024,tokens_per_block=16,num_warps=4,num_stages=2,page_size=16,quant_group_size=32,process_id=process,os_pid=os.getpid(),round_id=round_id,variant_order=ordinal,seed=seed,warmup=protocol['warmup'],repeat=protocol['repeat'],**env)
                    raw.append(row);write_csv(out/f'raw_p{process}.csv',raw)
            del fn,refs,ws,q,k,v,kp,ks,vp,vs,bt,lens
        except torch.OutOfMemoryError as e:
            fail.append(dict(context=t,batch=b,status='OOM',error=str(e)));(out/f'failures_p{process}.json').write_text(json.dumps(fail,indent=2))
        gc.collect();torch.cuda.empty_cache()
    if fail:raise RuntimeError('Incomplete sweep: OOM records retained')

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--worker',type=int);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.worker is not None:worker(a.out,a.worker,a.smoke);return
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');out=a.out or ROOT/f'kernel/results/E10_gqa_grouping/{stamp}_{os.getpid()}'
    out.mkdir(parents=True,exist_ok=False)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
    if dirty:raise RuntimeError('Tracked worktree must be clean')
    manifest=dict(experiment='E10_gqa_grouping',git_sha=sha,git_dirty=False,smoke=a.smoke,protocol=json.loads(PROTOCOL.read_text()),status='RUNNING',claim_eligible=False,created_utc=stamp,commands=[],note='Existing grouped path padded to power of two for Q_PER_KV=6; no nibble or merge changes')
    (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2));print('OUT',out,flush=True)
    with (out/'run.log').open('w') as log:
        for pid in range(1 if a.smoke else 3):
            cmd=[sys.executable,str(Path(__file__).resolve()),'--worker',str(pid),'--out',str(out)]+(['--smoke'] if a.smoke else [])
            manifest['commands'].append(cmd);(out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
            rc=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
            if rc:manifest.update(status='FAILED',return_code=rc);break
        else:manifest.update(status='COMPLETED',return_code=0)
    for pattern,name in [('raw_p*.csv','lowlevel_raw.csv'),('correctness_p*.csv','correctness.csv')]:
        rows=[]
        for f in sorted(out.glob(pattern)):rows+=list(csv.DictReader(f.open()))
        write_csv(out/name,rows)
    (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(manifest['status'],out,flush=True)
    if manifest['status']!='COMPLETED':raise RuntimeError('E10 failed; raw evidence retained')
if __name__=='__main__':main()
