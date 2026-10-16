"""E11 interventions; frozen E10 kernels. No performance tuning."""
import argparse,csv,gc,json,math,os,random,subprocess,sys,traceback
from pathlib import Path
from datetime import datetime,timezone
import torch,triton
from bench_gpu import cuda_ms
from bench_e10_grouping import error,write_csv
from ofd_kernel.reference.quant import quantize_int4_packed_groupwise
from ofd_kernel.runtime.launcher import TritonDecodeConfig,allocate_workspace
from ofd_kernel.runtime.subgrouped_launcher import subgrouped_paged_decode_core
ROOT=Path(__file__).resolve().parents[2]
CFG=TritonDecodeConfig(split_size=1024,tokens_per_block=16,num_warps=4,num_stages=2)

def cases(smoke):
    out=[]
    for t in ([2048] if smoke else [8192,32768]):
        for b in ([1] if smoke else [1,4]):
            for cap in [1,2,4]:out.append(dict(family='inactive_splits',T=t,B=b,Hkv=4,R=6,alias=1,cap=cap))
            for alias in [1,4,16]:out.append(dict(family='physical_reuse',T=t,B=b,Hkv=4,R=6,alias=alias,cap=1))
            for ratio in [1,2,3,6]:out.append(dict(family='q_repeat',T=t,B=b,Hkv=4,R=ratio,alias=1,cap=1))
    for b,h in ([(1,4),(2,2),(4,1)] if smoke else [(1,16),(2,8),(4,4),(8,2),(16,1)]):
        out.append(dict(family='batch_head_repartition',T=2048 if smoke else 32768,B=b,Hkv=h,R=6,alias=1,cap=1))
    return out

def build(c,seed):
    torch.manual_seed(seed);b,h,t,r=c['B'],c['Hkv'],c['T'],c['R'];n=b*t//16
    # Repartition uses identical repeated page/head values; alias uses exact copies.
    base=torch.randn(2,1,1,16,256,device='cuda',dtype=torch.float16)
    k=base[0].expand(n,h,16,256).contiguous();v=base[1].expand(n,h,16,256).contiguous()
    packed=[]
    for x in [k,v]:
        z=quantize_int4_packed_groupwise(x,32)
        packed.extend([z.packed.reshape(n,h,16,8,16),z.scale.reshape(n,h,16,8)])
    unique=n//c['alias'];ids=torch.arange(n,device='cuda')%unique
    # All capacities have a fully allocated valid block table; unused splits never read KV.
    bt=torch.zeros(b,t*c['cap']//16,device='cuda',dtype=torch.int32);bt[:,:t//16]=ids.reshape(b,-1).int()
    lens=torch.full((b,),t,device='cuda',dtype=torch.int32)
    qb=torch.randn(1,1,256,device='cuda',dtype=torch.float16)
    q=qb.expand(b,h*r,256).contiguous()
    return q,k,v,*packed,bt,lens

def worker(out,pid,smoke):
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();raw=[];checks=[]
    cs=cases(smoke);random.Random(9137+pid).shuffle(cs)
    env=dict(git_sha=sha,GPU=torch.cuda.get_device_name(),CUDA=torch.version.cuda,PyTorch=torch.__version__,Triton=triton.__version__)
    for c in cs:
        seed=9137+pid*1000 # same generated values within every intervention family
        print('CASE',pid,c,flush=True)
        q,k,v,kp,ks,vp,vs,bt,lens=build(c,seed);t,b,h,r=c['T'],c['B'],c['Hkv'],c['R'];cap=t*c['cap']
        ws=allocate_workspace(q,lens,CFG,max_seq_len=cap)
        def run(rep,g):return subgrouped_paged_decode_core(q,kp if rep=='INT4' else k,ks if rep=='INT4' else k,vp if rep=='INT4' else v,vs if rep=='INT4' else v,bt,lens,config=CFG,workspace=ws,max_seq_len=cap,representation=rep,group_q=g)
        variants=[(rep,g) for rep in ['INT4','FP16'] for g in [1,2,3,6] if r%g==0]
        errs={}
        # Explicit independent softmax oracle on one repeated page. All logical pages identical.
        for rep in ['INT4','FP16']:
            if rep=='FP16':kk=k[0,0].float();vv=v[0,0].float()
            else:
                def deq(pk,sc):
                    z=pk[0,0].int();codes=torch.stack([z&15,(z>>4)&15],-1).reshape(16,8,32)-8
                    return (codes.float()*sc[0,0].float()[...,None]).reshape(16,256)
                kk=deq(kp,ks);vv=deq(vp,vs)
            ref=((q[0,0].float()@kk.T/16).softmax(-1)@vv).expand_as(q)
            for rr,g in variants:
                if rr!=rep:continue
                y=run(rep,g);e=error(ref,y);finite=bool(torch.isfinite(y).all().item());errs[rep,g]=e
                checks.append(dict(**c,process_id=pid,representation=rep,group_q=g,relative_error=e,finite=finite,git_sha=sha))
                write_csv(out/f'correctness_p{pid}.csv',checks)
                if not finite or e>.002:raise RuntimeError('Oracle failure')
        ns=math.ceil(cap/1024);active=math.ceil(t/1024)
        for rnd in range(3):
            order=variants.copy();random.Random(seed+t+b+rnd).shuffle(order)
            for rep,g in order:
                ms=cuda_ms(lambda:run(rep,g),10,50)
                tensors=[kp,ks,vp,vs] if rep=='INT4' else [k,v]
                allocated=sum(x.numel()*x.element_size() for x in tensors)
                raw.append(dict(**c,representation=rep,group_q=g,process_id=pid,round_id=rnd,seed=seed,Hq=h*r,head_dim=256,capacity=cap,num_splits=ns,active_splits=active,W1=b*h*r*ns,stage1_program_count=b*h*(r//g)*ns,active_program_count=b*h*(r//g)*active,kv_allocated_bytes=allocated,kv_unique_address_bytes=allocated//c['alias'],workspace_bytes=sum(x.numel()*x.element_size() for x in [ws.partial_m,ws.partial_l,ws.partial_o,ws.out]),latency_ms=ms,aggregate_tok_s=b*1000/ms,query_heads_per_s=b*h*r*1000/ms,relative_error=errs[rep,g],split_size=1024,tokens_per_block=16,num_warps=4,num_stages=2,warmup=10,repeat=50,**env))
                write_csv(out/f'raw_p{pid}.csv',raw)
        del q,k,v,kp,ks,vp,vs,bt,lens,ws,ref,y,tensors
        gc.collect();torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--worker',type=int);p.add_argument('--out',type=Path);a=p.parse_args()
    if a.worker is not None:worker(a.out,a.worker,a.smoke);return
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise RuntimeError('Clean source required')
    out=ROOT/'kernel/results/E10_gqa_grouping'/('E11_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+str(os.getpid()));out.mkdir()
    manifest=dict(experiment='E11_controlled_interventions',git_sha=sha,status='RUNNING',smoke=a.smoke,cases=cases(a.smoke),independent_processes=1 if a.smoke else 3,rounds=3,protocol=json.loads((ROOT/'kernel/experiments/E11_controls/protocol.json').read_text()))
    (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2));print('OUT',out,flush=True)
    with (out/'run.log').open('w') as log:
        rc=0
        for pid in range(1 if a.smoke else 3):
            rc=subprocess.run([sys.executable,__file__,'--worker',str(pid),'--out',str(out)]+(['--smoke'] if a.smoke else []),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
            if rc:break
    manifest.update(status='COMPLETED' if rc==0 else 'FAILED',return_code=rc)
    for pat,name in [('raw_p*.csv','raw.csv'),('correctness_p*.csv','correctness.csv')]:
        rows=[]
        for f in sorted(out.glob(pat)):rows.extend(csv.DictReader(f.open()))
        write_csv(out/name,rows)
    (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest['status'],out,flush=True)
    if rc:raise RuntimeError('E11 failed; evidence preserved')
if __name__=='__main__':main()
