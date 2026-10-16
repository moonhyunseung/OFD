"""E12: exact-loop-guard intervention, old and disabled-guard controls."""
import argparse,csv,gc,json,math,os,random,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
import torch,triton
from bench_e10_grouping import cache,error,write_csv,CFG
from bench_gpu import cuda_ms
from ofd_kernel.runtime.launcher import allocate_workspace
from ofd_kernel.runtime.subgrouped_launcher import subgrouped_paged_decode_core
from ofd_kernel.runtime.guarded_launcher import guarded_paged_decode_core
ROOT=Path(__file__).resolve().parents[2]

def cases(smoke):
    cs=[dict(T=t,B=b,capacity=cap,ragged=False) for t in ([2048] if smoke else [8192,32768]) for b in ([1] if smoke else [1,4]) for cap in [t,t+16,t*4]]
    return cs+[dict(T=2048,B=4,capacity=cap,ragged=True) for cap in [2048,8192]]

def oracle(q,k,v,kp,ks,vp,vs,bt,lens,rep):
    outs=[]
    for b,L in enumerate(lens.cpu().tolist()):
        if L==0:outs.append(torch.zeros_like(q[b],dtype=torch.float32));continue
        ids=bt[b,:math.ceil(L/16)].long()
        def values(x,pk,sc):
            if rep=='FP16':return x[ids].permute(1,0,2,3).reshape(4,-1,256)[:,:L].float()
            z=pk[ids].int();codes=torch.stack([z&15,(z>>4)&15],-1).reshape(-1,4,16,8,32)-8
            y=(codes.float()*sc[ids].float()[...,None]).reshape(-1,4,16,256)
            return y.permute(1,0,2,3).reshape(4,-1,256)[:,:L]
        kk=values(k,kp,ks);vv=values(v,vp,vs);out=[]
        for h in range(24):out.append((q[b,h].float()@kk[h//6].T/16).softmax(-1)@vv[h//6])
        outs.append(torch.stack(out))
    return torch.stack(outs)

def worker(out,pid,smoke):
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();raw=[];checks=[];bank={}
    cs=cases(smoke);random.Random(12001+pid).shuffle(cs)
    for c in cs:
        t,b,cap=c['T'],c['B'],c['capacity'];seed=12001+pid*1000+t+b;key=(t,b)
        if key not in bank:bank[key]=cache(t,b,seed)
        q,k,v,kp,ks,vp,vs,orig_bt,orig_lens=bank[key]
        bt=torch.zeros(b,math.ceil(cap/16),device='cuda',dtype=torch.int32);bt[:,:t//16]=orig_bt
        lens=torch.tensor([0,1023,1024,1025],device='cuda',dtype=torch.int32) if c['ragged'] else orig_lens
        ws=allocate_workspace(q,lens,CFG,max_seq_len=cap)
        def run(rep,g,path):
            fn=subgrouped_paged_decode_core if path=='original' else guarded_paged_decode_core
            kw={} if path=='original' else dict(skip_empty=path=='skip')
            return fn(q,kp if rep=='INT4' else k,ks if rep=='INT4' else k,vp if rep=='INT4' else v,vs if rep=='INT4' else v,bt,lens,config=CFG,workspace=ws,max_seq_len=cap,representation=rep,group_q=g,**kw)
        variants=[(rep,g,path) for rep in ['INT4','FP16'] for g in [1,2,3,6] for path in ['original','disabled','skip']];errs={}
        for rep in ['INT4','FP16']:
            ref=run(rep,1,'original').clone()
            if smoke:
                expected=oracle(q,k,v,kp,ks,vp,vs,bt,lens,rep);e=error(expected,ref)
                if not torch.isfinite(ref).all().item() or e>.002:raise RuntimeError('Original oracle mismatch')
                checks.append(dict(**c,representation=rep,group_q=1,path='oracle',relative_error=e,finite=True,process_id=pid,git_sha=sha))
            for rr,g,path in variants:
                if rr!=rep:continue
                y=run(rep,g,path);e=error(ref,y);finite=bool(torch.isfinite(y).all().item())
                if c['ragged'] and not (y[0]==0).all().item():raise RuntimeError('Empty request not zero')
                errs[rep,g,path]=e
                checks.append(dict(**c,representation=rep,group_q=g,path=path,relative_error=e,finite=finite,process_id=pid,git_sha=sha));write_csv(out/f'correctness_p{pid}.csv',checks)
                if not finite or e>.002:raise RuntimeError('Guard correctness mismatch')
        for rnd in range(3):
            order=variants.copy();random.Random(seed+cap+pid*91+rnd).shuffle(order)
            for rep,g,path in order:
                ms=cuda_ms(lambda:run(rep,g,path),10,50);ns=math.ceil(cap/1024)
                xs=[kp,ks,vp,vs] if rep=='INT4' else [k,v]
                raw.append(dict(**c,representation=rep,group_q=g,path=path,process_id=pid,round_id=rnd,seed=seed,num_splits=ns,stage1_program_count=b*24//g*ns,active_program_count=sum(math.ceil(L/1024) for L in ([0,1023,1024,1025] if c['ragged'] else [t]*b))*24//g,latency_ms=ms,aggregate_tok_s=b*1000/ms,relative_error=errs[rep,g,path],kv_bytes=sum(x.numel()*x.element_size() for x in xs),kv_buffer_identity=':'.join(str(x.data_ptr()) for x in xs),Hq=24,Hkv=4,head_dim=256,split_size=1024,tokens_per_block=16,num_warps=4,num_stages=2,warmup=10,repeat=50,git_sha=sha,GPU=torch.cuda.get_device_name(),CUDA=torch.version.cuda,PyTorch=torch.__version__,Triton=triton.__version__))
                write_csv(out/f'raw_p{pid}.csv',raw)
        print('CASE_DONE',pid,c,flush=True);del ws,ref,y,bt,lens
        gc.collect();torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--worker',type=int);p.add_argument('--out',type=Path);a=p.parse_args()
    if a.worker is not None:worker(a.out,a.worker,a.smoke);return
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise RuntimeError('Clean source required')
    out=ROOT/'kernel/results/E10_gqa_grouping'/('E12_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+str(os.getpid()));out.mkdir()
    m=dict(git_sha=sha,experiment='E12_skip_empty',status='RUNNING',smoke=a.smoke,cases=cases(a.smoke),processes=1 if a.smoke else 3,rounds=3,protocol=json.loads((ROOT/'kernel/experiments/E12_skip_empty/protocol.json').read_text()))
    (out/'MANIFEST.json').write_text(json.dumps(m,indent=2));print('OUT',out,flush=True)
    with (out/'run.log').open('w') as log:
        rc=0
        for pid in range(m['processes']):
            rc=subprocess.run([sys.executable,__file__,'--worker',str(pid),'--out',str(out)]+(['--smoke'] if a.smoke else []),stdout=log,stderr=subprocess.STDOUT,cwd=ROOT).returncode
            if rc:break
    for pat,name in [('raw_p*.csv','raw.csv'),('correctness_p*.csv','correctness.csv')]:
        rs=[]
        for f in sorted(out.glob(pat)):rs.extend(csv.DictReader(f.open()))
        write_csv(out/name,rs)
    m.update(status='COMPLETED' if rc==0 else 'FAILED',return_code=rc);(out/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');print(m['status'],out,flush=True)
    if rc:raise RuntimeError('E12 failed; raw evidence retained')
if __name__=='__main__':main()
