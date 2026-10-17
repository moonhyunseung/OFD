"""Pure-Python analysis preparation; summaries retain strata and gate boundaries."""
import argparse,csv,json,statistics
from pathlib import Path

def rows(root,name):return list(csv.DictReader((root/(name+'.csv')).open()))
def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();root=a.run
 m=json.loads((root/'MANIFEST.json').read_text());result={'smoke':m['smoke'],'scientific_gate':'SMOKE_ONLY' if m['smoke'] else 'OBSERVATIONAL_NO_E15_ACCEPTANCE','source_sha':m['git_sha'],'block_frontiers':[],'support_frontiers':[],'plane_frontiers':[],'rank_windows':[]}
 bs=rows(root,'blocks')
 keys=['domain','context','layer','kv_head','basis','selection','blocks','heldout']
 grouped={}
 for x in bs:grouped.setdefault(tuple(x[k] for k in keys),[]).append(x)
 for key,rs in grouped.items():
  v=dict(zip(keys,key),n_steps=len(rs))
  for metric in ['energy_capture','score_relative_error','probability_TV_max']:
   vals=[float(x[metric]) for x in rs];v[metric+'_median']=statistics.median(vals);v[metric+'_min']=min(vals);v[metric+'_max']=max(vals)
  result['block_frontiers'].append(v)
 for table,out,axis,metrics in [('support','support_frontiers',['domain','context','layer','kv_head','target_mass'],['support_fraction','relative_error','N_eff']),('plane_frontier','plane_frontiers',['domain','context','layer','kv_head','basis','heldout'],['nonempty_planes_mean','contiguous_span_mean'])]:
  groups={}
  for x in rows(root,table):groups.setdefault(tuple(x[k] for k in axis),[]).append(x)
  for key,rs in groups.items():
   v=dict(zip(axis,key),n_rows=len(rs))
   for metric in metrics:
    vals=[float(x[metric]) for x in rs if x.get(metric) not in [None,'']]
    if vals:v[metric+'_median']=statistics.median(vals);v[metric+'_max']=max(vals)
   result[out].append(v)
 for x in rows(root,'rank'):
  result['rank_windows'].append({k:x[k] for k in ['domain','context','layer','kv_head','window','start','stable_rank','entropy_effective_rank','ranks']})
 (root/'summary.json').write_text(json.dumps(result,indent=2))
 report=['# E14 observational summary','',f"Source {m['git_sha']}; scientific gate: {result['scientific_gate']}.",'','Primary comparisons: frozen past-window basis on steps >=4 versus identity/H32/H256. Current-SVD and current-query block selection are oracle diagnostics. Rank<=6 is tautological; use window spectra and held-out operator error. Do not aggregate layer/head/sequence rows as independent prompts.','',f"Block strata: {len(result['block_frontiers'])}; support strata: {len(result['support_frontiers'])}; plane strata: {len(result['plane_frontiers'])}.",'','No automatic E15 gate: application error budgets, runtime legal basis, reprojection/rotation/scale/MMA work and generated code must be resolved first. WO metrics include the captured native output gate held fixed and measure group contributions, not propagated complete-model quality.','', 'See summary.json for per-domain/context/layer/head extrema and all raw CSV for individual steps.']
 (root/'FINDINGS.md').write_text('\n'.join(report)+'\n');print('E14_SUMMARY_READY',len(result['block_frontiers']))
if __name__=='__main__':main()
