"""Complete-model decode comparisons; preserve negative/no-crossover results."""
import argparse,csv,json,math,statistics
from collections import defaultdict
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();root=a.run
    manifest=json.loads((root/'MANIFEST.json').read_text());rows=list(csv.DictReader((root/'e2e.csv').open()))
    if manifest.get('smoke'):raise RuntimeError('Full run required')
    full=manifest['execution_status']=='COMPLETED'
    groups=defaultdict(list)
    for x in rows:
        if x['phase'] in ['evaluation','crossover']:groups[(x['phase'],x['domain'],int(x['context']),int(x['batch']))].append(x)
    table=[]
    for key,rs in sorted(groups.items()):
        baseline={int(x['round']):float(x['decode_ms']) for x in rs if x['policy_label']=='NATIVE'}
        paged={int(x['round']):float(x['decode_ms']) for x in rs if x['policy_label']=='FP16'}
        for label in ['NATIVE','FP16','P','KV','MIX']:
            xs=[x for x in rs if x['policy_label']==label]
            if not xs:continue
            ratios=[baseline[int(x['round'])]/float(x['decode_ms']) for x in xs if int(x['round']) in baseline]
            pr=[paged[int(x['round'])]/float(x['decode_ms']) for x in xs if int(x['round']) in paged]
            med=statistics.median(float(x['decode_ms']) for x in xs)
            table.append(dict(phase=key[0],domain=key[1],context=key[2],batch=key[3],policy=label,n=len(xs),median_decode_ms=med,aggregate_tok_s=key[3]*1000/med,native_speedup_median=statistics.median(ratios) if ratios else None,native_speedup_min=min(ratios) if ratios else None,native_speedup_max=max(ratios) if ratios else None,paged_fp16_speedup_median=statistics.median(pr) if pr else None,three_of_three_faster=len(ratios)==3 and min(ratios)>1,kv_bytes=int(xs[0]['kv_bytes']),allocated_bytes_median=statistics.median(int(x['allocated_bytes']) for x in xs),peak_bytes_max=max(int(x['peak_allocated_bytes']) for x in xs),logit_error_mean=statistics.mean(float(x['logit_rel_l2']) for x in xs),nll_mean=statistics.mean(float(x['nll']) for x in xs),argmax_agreement_mean=statistics.mean(float(x['argmax_agreement']) for x in xs)))
    if table:
        with (root/'e2e_summary.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    cal=defaultdict(list)
    for x in rows:
        if x['phase']=='calibration':cal[(int(x['changed_layer']),x['variant'])].append(float(x['decode_ms']))
    ct=[dict(layer=k[0],variant=k[1],n=len(v),geomean_e2e_ms=math.exp(statistics.mean(map(math.log,v))),min_ms=min(v),max_ms=max(v)) for k,v in sorted(cal.items())]
    if ct:
        with (root/'layer_4L_e2e.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(ct[0]));w.writeheader();w.writerows(ct)
    crossover=[]
    for b in sorted(set(x['batch'] for x in table if x['phase']=='crossover')):
        for label in ['P','KV','MIX']:
            xs=sorted((x for x in table if x['phase']=='crossover' and x['batch']==b and x['policy']==label),key=lambda x:x['context'])
            wins=[x for x in xs if x['three_of_three_faster']]
            crossover.append(dict(batch=b,policy=label,first_observed_context_with_3_of_3_wins=wins[0]['context'] if wins else None,measured_contexts=[x['context'] for x in xs],warning='Discrete observed point, not interpolated threshold; three replicates do not give a broad statistical guarantee'))
    policy_path=root/'speed_policy.json';policy=json.loads(policy_path.read_text()) if policy_path.exists() else None
    backend_log=(root/'run.log').read_text()
    reference_gdn=('falling back to its reference' in backend_log)
    result=dict(native_gdn_reference_fallback=reference_gdn,source_sha=manifest['git_commit'],complete=full,rows=len(rows),calibration_count=sum(len(v) for v in cal.values()),calibration_required=384,policy=policy,crossover=crossover,groups=table,failures=[json.loads(p.read_text()) for p in sorted(root.glob('OOM_*.json'))],scope='whole_model_teacher_forced_decode_including_online_KV_append',prefill_excluded=True,sampling_and_serving_excluded=True)
    (root/'e2e_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    text=['# Qwen3.8-27B complete-model decode','',f"Source `{manifest['git_commit']}`; run `{manifest['run_id']}`; completed={full}; rows={len(rows)}; 4L calibration measurements={result['calibration_count']}/384.",'','Primary latency is synchronized wall time for full model forwards after warmup. It includes all layers, online KV append, GPU FWHT and LM head. Prefill/setup, CPU sampling, network and request scheduling are excluded. Native FP16 uses the original SDPA attention with a preallocated dense cache, checked against unmodified cache in smoke. Paged FP16 uses the same page table/split principles as INT4. Full-attention FP16 shadow caches are not retained in INT4 decode. The base template is held on CPU.','',f"Selected speed policy (frozen before held-out evaluation): {policy}",'','| Phase | Domain | Context | B | Policy | Median E2E ms | Native speedup | Paged FP16 speedup | n |','|---|---|---:|---:|---|---:|---:|---:|---:|']
    for x in table:
        ns='NA' if x['native_speedup_median'] is None else f"{x['native_speedup_median']:.4f}"
        ps='NA' if x['paged_fp16_speedup_median'] is None else f"{x['paged_fp16_speedup_median']:.4f}"
        text.append(f"| {x['phase']} | {x['domain']} | {x['context']} | {x['batch']} | {x['policy']} | {x['median_decode_ms']:.4f} | {ns} | {ps} | {x['n']} |")
    text+=['','## Limits',f'Native Gated DeltaNet reference fallback observed in run.log: {reference_gdn}. All compared paths use the same native linear-attention code. These timings do not establish performance against an optimized serving stack with causal-conv1d/flash-linear-attention fast paths.','The 4L perturbations are measured around all-P, and cannot establish an exhaustive 4^16 optimum. The combined policy is validated separately on held-out streams. Repeated rounds share workload/model; they are not independent process replicas. Short teacher-forced continuations provide logit/NLL diagnostics, not benchmark task-quality or free-generation guarantees. Replicated batch inputs use independently allocated KV and no prefix sharing, but do not cover heterogeneous request content. KV representation compression is distinct from total GPU memory reduction. Host-template limits and prefill/decode setup OOMs are preserved and are not evidence of an INT4 residency frontier. A speed crossover is reported only at measured shapes; no unmeasured threshold is inferred. Native static-cache SDPA is the main real-model comparator; matched paged FP16 is reported separately.','',f"Observed crossover entries: {crossover}"]
    (root/'E2E_REPORT.md').write_text('\n'.join(text)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['groups','policy']},indent=2))
if __name__=='__main__':main()
