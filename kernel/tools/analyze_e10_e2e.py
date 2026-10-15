"""E10 model confirmation: paired g1/frozen-g effects, no new selection."""
import argparse,csv,json,math,statistics,hashlib
from pathlib import Path

def analyze(root):
    m=json.loads((root/'MANIFEST.json').read_text());rows=list(csv.DictReader((root/'e2e.csv').open()));smoke=m['smoke'];expected=12 if smoke else 96
    grouped={}
    expected_shapes={(128,1)} if smoke else {tuple(v) for v in m['policy']['fullmodel_shapes']}
    for x in rows:
        k=(int(x['context']),int(x['batch']),x['representation'],int(x['round']),x['policy_label'])
        if (k[0],k[1]) not in expected_shapes:raise RuntimeError('Unexpected model shape')
        if x['status']!='OK' or not math.isfinite(float(x['decode_ms'])) or float(x['decode_ms'])<=0:raise RuntimeError('Invalid model timing/status')
        if k in grouped:raise RuntimeError('Duplicate model row')
        if x['git_sha']!=m['git_sha']:raise RuntimeError('Source mismatch')
        g=int(x['group_q']);exp=1 if x['policy_label'].endswith('_g1') else m['policy']['fixed'][x['representation']]
        if g!=exp:raise RuntimeError('Frozen policy changed')
        if int(x['capacity'])!=int(x['context'])+(4 if smoke else 16):raise RuntimeError('Capacity mismatch')
        if int(x['num_splits'])!=(int(x['capacity'])+1023)//1024:raise RuntimeError('Ns mismatch')
        if int(x['stage1_program_count'])!=int(x['batch'])*4*(6//g)*int(x['num_splits']):raise RuntimeError('Stage1 work mismatch')
        if int(x['W1'])!=int(x['batch'])*24*int(x['num_splits']):raise RuntimeError('W1 mismatch')
        grouped[k]=x
    table=[]
    for t,b in sorted(set((k[0],k[1]) for k in grouped)):
        for rep in ['FP16','INT4']:
            a=[grouped[t,b,rep,r,rep+'_g1'] for r in range(3) if (t,b,rep,r,rep+'_g1') in grouped]
            z=[grouped[t,b,rep,r,rep+'_gbest'] for r in range(3) if (t,b,rep,r,rep+'_gbest') in grouped]
            pairs=[(grouped[t,b,rep,r,rep+'_g1'],grouped[t,b,rep,r,rep+'_gbest']) for r in range(3) if (t,b,rep,r,rep+'_g1') in grouped and (t,b,rep,r,rep+'_gbest') in grouped]
            if not pairs:continue
            ratios=[float(x['decode_ms'])/float(y['decode_ms']) for x,y in pairs];deltas=[float(x['decode_ms'])-float(y['decode_ms']) for x,y in pairs]
            med_a=statistics.median(float(x['decode_ms']) for x in a);med_z=statistics.median(float(x['decode_ms']) for x in z)
            table.append(dict(context=t,batch=b,representation=rep,frozen_group_q=m['policy']['fixed'][rep],n_pairs=len(pairs),g1_median_ms=med_a,gbest_median_ms=med_z,g1_tok_s=b*1000/med_a,gbest_tok_s=b*1000/med_z,paired_speedup_median=statistics.median(ratios),paired_speedup_min=min(ratios),paired_speedup_max=max(ratios),paired_gain_ms_median=statistics.median(deltas),paired_gain_ms_min=min(deltas),paired_gain_ms_max=max(deltas),g1_kv_bytes=int(a[0]['kv_bytes']),gbest_kv_bytes=int(z[0]['kv_bytes']),g1_allocated_bytes_median=statistics.median(int(x['allocated_bytes']) for x in a),gbest_allocated_bytes_median=statistics.median(int(x['allocated_bytes']) for x in z),g1_gpu_event_ms_median=statistics.median(float(x['gpu_event_ms']) for x in a),gbest_gpu_event_ms_median=statistics.median(float(x['gpu_event_ms']) for x in z),num_splits=int(a[0]['num_splits']),git_sha=m['git_sha']))
    if table:
        with (root/'confirmation_summary.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    failures=[dict(path=p.name,**json.loads(p.read_text())) for p in root.glob('OOM_*.json')]
    result=dict(source_sha=m['git_sha'],complete=m['status']=='COMPLETED' and len(rows)==expected and not failures,rows=len(rows),expected_rows=expected,smoke=smoke,policy=m['policy'],failures=failures,groups=table,native_gdn_reference_fallback='falling back to its reference' in (root/'run.log').read_text())
    (root/'confirmation_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    text=['# E10 whole-model grouping confirmation','',f"Source {m['git_sha']}; complete={result['complete']}; rows={len(rows)}/{expected}; smoke={smoke}.",'','Fixed grouping policy was committed before any model run. FP16 best fixed g=1, so FP16_gbest repeats the g1 execution as a timing control; it is not a distinct optimized kernel. INT4 uses plain P with g=2. No calibration or layer selection is performed.','', '| Context | B | Representation | Fixed g | g1 ms | Fixed-g ms | Paired speedup median | Paired speedup range |','|---|---:|---|---:|---:|---:|---:|---|']
    for x in table:text.append(f"| {x['context']} | {x['batch']} | {x['representation']} | {x['frozen_group_q']} | {x['g1_median_ms']:.4f} | {x['gbest_median_ms']:.4f} | {x['paired_speedup_median']:.4f} | {x['paired_speedup_min']:.4f} .. {x['paired_speedup_max']:.4f} |")
    text+=['','All64 layers, online append, LM head and teacher-forced decode are timed using unchanged E09 wall/event timing. Full runs use 16 decode steps, four warmup and twelve timed; smoke uses four steps and one warmup. Prefill, setup, sampling and serving scheduling are excluded. Online capacity is T+decode_steps, Ns=ceil((T+decode_steps)/1024), unlike standalone low-level exact T. Fixed-g comparisons use the same capacity, layout, bytes and model/data revision.','',f"Native GDN reference fallback observed: {result['native_gdn_reference_fallback']}. This does not establish performance with optimized native fast paths or production serving. Rounds share one model process; small speed differences are not independent-process confirmation. FP16 gbest/g1 is an identical-execution control: its observed round spread must be considered before attributing small whole-model effects to grouping. No native logit reference is collected in this speed-only confirmation; finite logits are required and online subgroup equivalence is checked separately in the smoke.",'',f'Failures/skips: {failures}']
    (root/'CONFIRMATION.md').write_text('\n'.join(text)+'\n');return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();r=analyze(a.run);print(json.dumps({k:v for k,v in r.items() if k not in ['groups','policy']},indent=2))
