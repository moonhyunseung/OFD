"""Summarize completed decode replay without inferring serving performance."""
import argparse,csv,json,math,statistics
from collections import Counter
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();d=a.run
    manifest=json.loads((d/'MANIFEST.json').read_text());summary=json.loads((d/'summary.json').read_text())
    if manifest['execution_status']!='COMPLETED' or summary['smoke']:raise RuntimeError('Completed full replay required')
    rows=list(csv.DictReader((d/'layer_decode.csv').open()));policy=json.loads((d/'policy.json').read_text())['policy']
    selected=[x for x in rows if x['phase']=='evaluation' and x['variant']==policy[x['layer']]]
    plain=[x for x in rows if x['phase']=='evaluation' and x['variant']=='P']
    def mean(rs,key):return statistics.mean(float(x[key]) for x in rs)
    def geo(rs,key):return math.exp(mean([dict(v=math.log(float(x[key]))) for x in rs],'v'))
    fields=['layer','policy','n_eval','plain_error_mean','selected_error_mean','selected_vs_plain_error_ratio','fp16_over_selected_latency_geomean']
    layers=[]
    for layer in sorted(policy,key=int):
        ss=[x for x in selected if x['layer']==layer];pp=[x for x in plain if x['layer']==layer]
        if len(ss)!=len(pp) or not ss:raise RuntimeError('Unmatched evaluation coverage')
        layers.append(dict(layer=int(layer),policy=policy[layer],n_eval=len(ss),plain_error_mean=mean(pp,'relative_l2_error'),selected_error_mean=mean(ss,'relative_l2_error'),selected_vs_plain_error_ratio=mean(ss,'relative_l2_error')/mean(pp,'relative_l2_error'),fp16_over_selected_latency_geomean=geo(ss,'throughput_ratio')))
    with (d/'layer_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(layers)
    groups=[]
    for domain,context in sorted(set((x['domain'],int(x['context'])) for x in selected)):
        ss=[x for x in selected if x['domain']==domain and int(x['context'])==context]
        groups.append(dict(domain=domain,context=context,n=len(ss),selected_error_mean=mean(ss,'relative_l2_error'),fp16_over_selected_latency_geomean=geo(ss,'throughput_ratio')))
    result=dict(git_commit=manifest['git_commit'],run_id=manifest['run_id'],policy_counts=dict(Counter(policy.values())),layers=layers,domain_context=groups,mean_selected_error=mean(selected,'relative_l2_error'),mean_plain_error=mean(plain,'relative_l2_error'),fp16_over_selected_latency_geomean=geo(selected,'throughput_ratio'),scope='real_model_decode_layer_replay_only',e09_serving_complete=False)
    (d/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    text=['# Qwen3.8-27B decode layer replay','',f"Source: `{manifest['git_commit']}`; run `{manifest['run_id']}`.",'','Prefill excluded; 16 full-attention layers, four domains, fixed calibration/evaluation partition. Other 48 Gated DeltaNet layers remain native. Policy chosen from calibration error only; all figures below use evaluation.','',f"Policy counts: {result['policy_counts']}. Mean attention-output relative error: P={result['mean_plain_error']:.6g}, selected={result['mean_selected_error']:.6g}. Geometric mean matched paged FP16/selected-INT4 latency ratio={result['fp16_over_selected_latency_geomean']:.6g}.",'','| Layer | Policy | P error | Selected error | Selected/P error | FP16/INT4 latency |','|---:|---|---:|---:|---:|---:|']
    for x in layers:text.append(f"| {x['layer']} | {x['policy']} | {x['plain_error_mean']:.5g} | {x['selected_error_mean']:.5g} | {x['selected_vs_plain_error_ratio']:.4f} | {x['fp16_over_selected_latency_geomean']:.4f} |")
    text+=['','Direct measurements concern per-layer attention replay on native model decode states. The selected policy is replay-evaluated, not installed in the model. Total model latency, online KV append, generated-text quality, perplexity and resident serving capacity are unmeasured. One packed stream per domain/partition and nested context prefixes limit statistical generalization. GPU events use the inherited host-launch timing contract and include possible launch starvation. No hardware-counter diagnosis is implied.','', 'System interpretation: lower selected error demonstrates held-out attention-output improvement only where measured. It does not establish downstream task quality or full-model speedup. Next required step is online packed-cache integration and an actual policy-conditioned decode comparison.']
    (d/'REPORT.md').write_text('\n'.join(text)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['layers','domain_context']},indent=2))
if __name__=='__main__':main()
