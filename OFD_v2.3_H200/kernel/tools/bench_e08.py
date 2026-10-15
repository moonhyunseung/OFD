"""Frozen H200 launch tuning; global calibration selection then held-out cases."""
import argparse,csv,gc,itertools,json,math,random,statistics
from pathlib import Path
import torch
from bench_gpu import cuda_ms,rel
from bench_e03 import make_random_paged_case
from ofd_kernel.bench import BenchmarkCase
from ofd_kernel.runtime.launcher import TritonDecodeConfig,allocate_workspace,triton_paged_decode_core
ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if torch.cuda.get_device_capability()!=(9,0) or 'H200' not in torch.cuda.get_device_name(): raise RuntimeError('H200 target required')
    contract=json.loads((ROOT/'kernel/experiments/E08_h200_tuning/config.json').read_text())
    configs=[TritonDecodeConfig(*x) for x in itertools.product([512,1024,2048],[16,32],[4,8],[1,2])]
    base=TritonDecodeConfig(); bi=configs.index(base)
    if a.smoke: configs=[base,TritonDecodeConfig(512,32,4,1)];bi=0
    def case(t,b,r,ids,phase,writer,f):
        seed=1337+t+b+r
        q,kp,ks,vp,vs,bt,lens=make_random_paged_case(BenchmarkCase(t,b,1),num_kv_heads=8,device=torch.device('cuda'),seed=seed)
        q=torch.randn(b,8*r,128,device='cuda',dtype=torch.float16)
        work={i:allocate_workspace(q,lens,configs[i],max_seq_len=t) for i in ids}
        def fn(i):return triton_paged_decode_core(q,kp,ks,vp,vs,bt,lens,config=configs[i],workspace=work[i],max_seq_len=t)
        ref=fn(bi).clone();torch.cuda.synchronize();errors={}
        for i in ids:
            err=rel(fn(i),ref);errors[i]=err
            if not math.isfinite(err) or err>0.002:raise RuntimeError(f'correctness cfg={i} error={err}')
        measured={i:[] for i in ids}
        for rnd in range(1 if a.smoke else 3):
            order=list(ids);random.Random(seed+rnd).shuffle(order)
            for i in order:
                ms=cuda_ms(lambda:fn(i),5,30);measured[i].append(ms)
                row=dict(phase=phase,context=t,batch=b,q_per_kv=r,config_id=i,round=rnd,latency_ms=ms,tok_s=b*1000/ms,relative_error=errors[i],seed=seed,**configs[i].__dict__)
                writer.writerow(row);f.flush()
        med={i:statistics.median(v) for i,v in measured.items()}
        print(phase,t,b,r,'baseline_ms',med[bi],'best_speedup',max(med[bi]/v for v in med.values()),flush=True)
        return {i:med[bi]/v for i,v in med.items()}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    fields=['phase','context','batch','q_per_kv','config_id','round','latency_ms','tok_s','relative_error','seed','split_size','tokens_per_block','num_warps','num_stages']
    scores={i:[] for i in range(len(configs))}; evaluation=[]
    with a.out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for t,b,r in ([(1024,1,1)] if a.smoke else contract['calibration']):
            ratios=case(t,b,r,list(scores),'calibration',w,f)
            for i,v in ratios.items():scores[i].append(v)
            gc.collect();torch.cuda.empty_cache()
        geo={i:math.exp(sum(map(math.log,v))/len(v)) for i,v in scores.items()}
        selected=max(geo,key=geo.get)
        selection=dict(selected_config_id=selected,selected_config=configs[selected].__dict__,baseline_config_id=bi,calibration_geomean=geo[selected],selection_frozen_before_evaluation=True)
        (a.out.parent/'selection.json').write_text(json.dumps(selection,indent=2)+'\n');print('FROZEN_SELECTION',selection,flush=True)
        for t,b,r in ([(2048,1,4)] if a.smoke else contract['evaluation']):
            ratios=case(t,b,r,sorted(set([bi,selected])),'evaluation',w,f);evaluation.append(dict(context=t,batch=b,q_per_kv=r,speedup=ratios[selected]))
            gc.collect();torch.cuda.empty_cache()
    (a.out.parent/'summary.json').write_text(json.dumps(dict(**selection,evaluation=evaluation,evaluation_geomean=math.exp(sum(math.log(x['speedup']) for x in evaluation)/len(evaluation)),smoke=a.smoke,scope='core_only',profiling_counters_collected=False),indent=2)+'\n')
if __name__=='__main__':main()
