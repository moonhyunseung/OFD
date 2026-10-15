"""Frozen E04b GPU-FWHT end-to-end attention and E06 grouped reuse sweeps."""
import argparse,csv,gc,math
from pathlib import Path
import torch
from bench_gpu import cuda_ms,rel
from bench_e03 import make_random_paged_case
from ofd_kernel.bench import BenchmarkCase
from ofd_kernel.contracts import Variant
from ofd_kernel.runtime.launcher import TritonDecodeConfig,allocate_workspace,triton_paged_decode_core
from ofd_kernel.runtime.grouped_launcher import triton_paged_decode_grouped_core
from ofd_kernel.runtime.gpu_ofd import gpu_ofd_decode
CFG=TritonDecodeConfig()


def one(eid,t,b,r,warmup,repeat):
    seed=1337+t+b+r
    q,kp,ks,vp,vs,bt,lens=make_random_paged_case(BenchmarkCase(t,b,1),num_kv_heads=8,device=torch.device('cuda'),seed=seed)
    q=torch.randn(b,8*r,128,device='cuda',dtype=torch.float16)
    ws=allocate_workspace(q,lens,CFG,max_seq_len=t)
    rows=[]
    if eid=='E06_gqa':
        wg=allocate_workspace(q,lens,CFG,max_seq_len=t)
        def plain():return triton_paged_decode_core(q,kp,ks,vp,vs,bt,lens,config=CFG,workspace=ws,max_seq_len=t)
        def grouped():return triton_paged_decode_grouped_core(q,kp,ks,vp,vs,bt,lens,config=CFG,workspace=wg,max_seq_len=t)
        plain();grouped();torch.cuda.synchronize()
        err=rel(ws.out,wg.out)
        if not math.isfinite(err) or err>0.002:raise RuntimeError(f'Grouped matrix correctness {err}')
        pm=cuda_ms(plain,warmup,repeat);gm=cuda_ms(grouped,warmup,repeat)
        rows=[dict(variant='P',baseline_ms=pm,target_ms=gm,baseline_tok_s=b*1000/pm,target_tok_s=b*1000/gm,target_baseline_throughput_ratio=pm/gm,relative_error=err,timing_scope='core_attention_only')]
    else:
        qw=torch.empty_like(q);ow=torch.empty_like(q)
        for variant in Variant:
            def fn():return gpu_ofd_decode(q,kp,ks,vp,vs,bt,lens,variant=variant,config=CFG,workspace=ws,max_seq_len=t,q_work=qw,out_work=ow)
            output=fn();torch.cuda.synchronize()
            if not torch.isfinite(output).all().item():raise RuntimeError('Non-finite GPU OFD output')
            ms=cuda_ms(fn,warmup,repeat)
            rows.append(dict(variant=variant.value,target_ms=ms,target_tok_s=b*1000/ms,timing_scope='gpu_fwht_plus_attention_plus_gpu_fwht'))
        p=next(x['target_ms'] for x in rows if x['variant']=='P')
        for row in rows:
            row.update(baseline_ms=p,baseline_tok_s=b*1000/p,target_baseline_throughput_ratio=p/row['target_ms'])
    for row in rows:
        row.update(context=t,batch=b,q_per_kv=r,hkv=8,head_dim=128,page_size=16,group_size=32,dtype='fp16',seed=seed,split_size=1024,tokens_per_block=16,num_warps=4,num_stages=2,status='OK')
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',choices=['E04b_end_to_end','E06_gqa'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.smoke:matrix=[(8192,1,r) for r in ([1,4] if a.experiment=='E04b_end_to_end' else [1,4,8])]
    elif a.experiment=='E04b_end_to_end':matrix=[(t,b,r) for t in [8192,32768,65536,131072] for b in [1,4,8,16] for r in [1,4]]
    else:matrix=[(t,b,r) for t in [8192,32768,131072] for b in [1,4,8,16] for r in [1,4,8]]
    fields=['context','batch','q_per_kv','hkv','head_dim','page_size','group_size','dtype','variant','baseline_ms','target_ms','baseline_tok_s','target_tok_s','target_baseline_throughput_ratio','relative_error','timing_scope','seed','split_size','tokens_per_block','num_warps','num_stages','status']
    worst=0;oom=0
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for idx,(t,b,r) in enumerate(matrix):
            try:
                rows=one(a.experiment,t,b,r,3,10)
            except torch.OutOfMemoryError:
                rows=[dict(context=t,batch=b,q_per_kv=r,status='OOM')];oom+=1
            for row in rows:
                w.writerow(row);f.flush()
                if row.get('variant')=='KV':worst=max(worst,row['target_ms']/row['baseline_ms'])
            print(idx+1,len(matrix),t,b,r,[(x.get('variant'),x.get('target_baseline_throughput_ratio'),x.get('status')) for x in rows],flush=True)
            gc.collect();torch.cuda.empty_cache()
    if a.experiment=='E04b_end_to_end':
        print('FROZEN_E04B_WORST_KV_P_RATIO',worst,'THRESHOLD',1.05,flush=True)
        if oom or (not a.smoke and worst>1.05):raise RuntimeError('E04b performance gate failed; complete raw matrix retained')
    elif oom:raise RuntimeError('E06 matrix contains OOM; incomplete gate')
if __name__=='__main__':main()
