"""Analyze E10 paired effects, retaining raw process variation and equal-work contrasts."""
import argparse,csv,json,math,statistics,hashlib
from pathlib import Path
from collections import defaultdict

def dump_csv(p,rs):
    if rs:
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)

def analyze(root):
    manifest=json.loads((root/'MANIFEST.json').read_text());protocol=manifest['protocol']
    raw=list(csv.DictReader((root/'lowlevel_raw.csv').open()));checks=list(csv.DictReader((root/'correctness.csv').open()))
    expected=(1 if manifest['smoke'] else 24)*10*protocol['rounds']*(1 if manifest['smoke'] else protocol['processes'])
    complete=manifest['status']=='COMPLETED' and len(raw)==expected
    correct=all(x['finite']=='True' and float(x['relative_error'])<=.002 for x in checks)
    data={}
    for x in raw:
        key=(int(x['context']),int(x['batch']),x['representation'],int(x['group_q']),int(x['process_id']),int(x['round_id']),x['path'])
        if key in data:raise RuntimeError('Duplicate measurement key')
        value=float(x['latency_ms'])
        if not math.isfinite(value) or value <= 0:raise RuntimeError('Invalid latency')
        if x['git_sha']!=manifest['git_sha']:raise RuntimeError('Raw source SHA mismatch')
        data[key]=value
    shapes=sorted(set((k[0],k[1]) for k in data));pr=sorted(set((k[4],k[5]) for k in data))
    summary=[];sanity=[];effects=[];surface=[];eq=[]
    for t,b in shapes:
        for rep in ['INT4','FP16']:
            for g in [1,2,3,6]:
                xs=[v for k,v in data.items() if k[:4]==(t,b,rep,g) and k[-1]=='generalized'];med=statistics.median(xs)
                summary.append(dict(context=t,batch=b,representation=rep,group_q=g,n=len(xs),latency_ms=med,latency_min=min(xs),latency_max=max(xs),aggregate_tok_s=b*1000/med,num_splits=(t+1023)//1024,W1=b*24*((t+1023)//1024),stage1_program_count=b*24*((t+1023)//1024)//g))
            for pid,rnd in pr:
                gen=data.get((t,b,rep,1,pid,rnd,'generalized'));old=data.get((t,b,rep,1,pid,rnd,'legacy'))
                if gen is not None and old is not None:sanity.append(dict(context=t,batch=b,representation=rep,process_id=pid,round_id=rnd,generalized_ms=gen,legacy_ms=old,ratio=gen/old))
            ss=[x for x in summary if (x['context'],x['batch'],x['representation'])==(t,b,rep)]
            best=min(ss,key=lambda x:(x['latency_ms'],x['group_q']))
            surface.append(dict(context=t,batch=b,representation=rep,g_star=best['group_q'],latency_ms=best['latency_ms'],W1=best['W1']))
        for g in [2,3,6]:
            for pid,rnd in pr:
                keys=[(t,b,rep,gg,pid,rnd,'generalized') for rep,gg in [('INT4',1),('INT4',g),('FP16',1),('FP16',g)]]
                if all(k in data for k in keys):
                    i1,ig,f1,fg=[data[k] for k in keys];gi=i1-ig;gf=f1-fg
                    effects.append(dict(context=t,batch=b,group_q=g,process_id=pid,round_id=rnd,GI_ms=gi,GF_ms=gf,D_ms=gi-gf,INT4_speedup=i1/ig,FP16_speedup=f1/fg))
    effectsummary=[]
    for t,b in shapes:
        for g in [2,3,6]:
            xs=[x for x in effects if (x['context'],x['batch'],x['group_q'])==(t,b,g)]
            if not xs:continue
            z=dict(context=t,batch=b,group_q=g,n=len(xs))
            for key in ['GI_ms','GF_ms','D_ms','INT4_speedup','FP16_speedup']:
                vs=[x[key] for x in xs];z[key+'_median']=statistics.median(vs);z[key+'_min']=min(vs);z[key+'_max']=max(vs)
            # Three process means are independent units; rounds are not independent processes.
            pm=[statistics.mean(x['D_ms'] for x in xs if x['process_id']==pid) for pid in sorted(set(x['process_id'] for x in xs))]
            z['D_process_mean_min']=min(pm);z['D_process_mean_max']=max(pm);effectsummary.append(z)
    for gid,group in enumerate(protocol['equal_work']):
        ref_t,ref_b=group[0]
        for t,b in group:
            for rep in ['INT4','FP16']:
                for g in [1,2,3,6]:
                    contrasts=[];gains=[]
                    for pid,rnd in pr:
                        keys=[(t,b,rep,g,pid,rnd,'generalized'),(ref_t,ref_b,rep,g,pid,rnd,'generalized'),(t,b,rep,1,pid,rnd,'generalized')]
                        if all(k in data for k in keys):
                            val,ref,one=[data[k] for k in keys];contrasts.append(val/ref);gains.append(one-val)
                    if contrasts:eq.append(dict(equal_work_group=gid,context=t,batch=b,representation=rep,group_q=g,W1=b*24*((t+1023)//1024),reference_context=ref_t,reference_batch=ref_b,latency_ratio_to_reference=statistics.median(contrasts),ratio_min=min(contrasts),ratio_max=max(contrasts),grouping_gain_ms=statistics.median(gains)))
    for name,rows in [('lowlevel_summary.csv',summary),('g1_sanity.csv',sanity),('grouping_effects_raw.csv',effects),('grouping_effects.csv',effectsummary),('grouping_surface.csv',surface),('equal_work.csv',eq)]:dump_csv(root/name,rows)
    fixed={}
    if complete and correct and not manifest['smoke']:
        for rep in ['INT4','FP16']:
            values={g:math.exp(statistics.mean(math.log(v) for k,v in data.items() if k[2]==rep and k[3]==g and k[-1]=='generalized')) for g in [1,2,3,6]}
            fixed[rep]=dict(group_q=min(values,key=lambda g:(values[g],g)),geomean_latency_ms=values)
    result=dict(source_sha=manifest['git_sha'],complete=complete,correctness_pass=correct,rows=len(raw),expected_rows=expected,smoke=manifest['smoke'],g1_ratio_range={rep:[min(x['ratio'] for x in sanity if x['representation']==rep),max(x['ratio'] for x in sanity if x['representation']==rep)] for rep in ['INT4','FP16']},fixed_choice_candidate=fixed,fullmodel_gate='PENDING_MANUAL_G1_SANITY_REVIEW',effects=effectsummary)
    (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    text=['# E10 GQA grouping causal sweep','',f"Measurement SHA: {manifest['git_sha']}; complete={complete}; correctness={correct}; rows={len(raw)}/{expected}.",'','## 1. Direct observations','See lowlevel_summary.csv (median/min/max of nine process/round blocks) and g1_sanity.csv. Repeats inside each block are not independent experimental units.','',f"Generalized/legacy g1 ratios: {result['g1_ratio_range']}",'','## 2. INT4 grouping gain','GI(g)=INT4_g1-INT4_g, in ms. Positive means lower latency.','', '## 3. FP16 grouping gain','GF(g)=FP16_g1-FP16_g, in ms. The identical grouping and padding control generic granularity effects.','','## 4. Difference-in-differences D(g)','D=GI-GF, computed within each process/round before aggregation. Raw effects and process mean ranges are retained. No posthoc significance threshold.','','| T | B | g | GI median ms | GF median ms | D median ms | D process-mean range ms |','|---|---:|---:|---:|---:|---:|---|']
    for x in effectsummary:text.append(f"| {x['context']} | {x['batch']} | {x['group_q']} | {x['GI_ms_median']:.6f} | {x['GF_ms_median']:.6f} | {x['D_ms_median']:.6f} | {x['D_process_mean_min']:.6f} .. {x['D_process_mean_max']:.6f} |")
    text+=['','## 5. Equal-work batch analysis','See equal_work.csv. All requested triplets use exact T capacity and have equal W1. Each ratio is paired by process/round against the first triplet member. Context, split count and batch change together; same W1 also means same B*T and KV footprint.','','## 6. Supported interpretation','Pending inspection of measured effects and g1 compatibility; no conclusion is generated from completion alone. Grouping reduces program count and within-subgroup repeated representation work simultaneously. D estimates representation-dependent benefit, not an isolated instruction count.','','## 7. Rejected interpretation','This test cannot establish exact HBM traffic, spills, instruction percentages, total-work-only causality, or pure batch causality independent of context. Do not claim generalized behavior from a smoke.','','## 8. Remaining uncertainty','Process variation, padding lanes (3->4,6->8), shared merge floor, cache residency, and compiler/resource changes remain. Matched FP16 controls do not remove all representation-by-scheduling interactions. No full-model confirmation until manual validity review and a committed fixed policy.','',f'Best fixed g candidates (not yet approved/frozen for model): {fixed}']
    (root/'FINDINGS.md').write_text('\n'.join(text)+'\n');return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();r=analyze(a.run);print(json.dumps({k:v for k,v in r.items() if k!='effects'},indent=2))
