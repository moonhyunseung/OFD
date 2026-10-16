"""E12 paired original/disabled/enabled effects and exact measurement validation."""
import argparse,csv,json,math,statistics,hashlib
from pathlib import Path

def write(p,rs):
    if not rs:return
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)

def analyze(root):
    m=json.loads((root/'MANIFEST.json').read_text());xs=list(csv.DictReader((root/'raw.csv').open()));cs=list(csv.DictReader((root/'correctness.csv').open()));axes=['T','B','capacity','ragged','representation','group_q'];data={};identities={}
    for x in xs:
        k=tuple(x[a] for a in axes+['process_id','round_id','path'])
        if k in data:raise RuntimeError('Duplicate timing')
        assert x['git_sha']==m['git_sha'];v=float(x['latency_ms']);assert v>0 and math.isfinite(v);data[k]=v
        ns=(int(x['capacity'])+1023)//1024;assert ns==int(x['num_splits']);assert int(x['stage1_program_count'])==int(x['B'])*24//int(x['group_q'])*ns
        identities.setdefault((x['T'],x['B'],x['representation'],x['process_id']),set()).add(x['kv_buffer_identity'])
    assert all(len(v)==1 for v in identities.values()),'KV pointer control failed'
    paired=[];summary=[]
    for key in sorted({k[:-1] for k in data}):
        o,d,s=[data[key+(mode,)] for mode in ['original','disabled','skip']]
        paired.append(dict(zip(axes+['process_id','round_id'],key),original_ms=o,disabled_ms=d,skip_ms=s,disabled_over_original=d/o,original_over_skip=o/s,disabled_over_skip=d/s,saved_ms=d-s))
    for key in sorted({tuple(x[a] for a in axes) for x in paired}):
        rs=[x for x in paired if tuple(x[a] for a in axes)==key];row=dict(zip(axes,key),n=len(rs))
        for stat in ['original_ms','disabled_ms','skip_ms','disabled_over_original','original_over_skip','disabled_over_skip','saved_ms']:
            vals=[x[stat] for x in rs];means=[statistics.mean(x[stat] for x in rs if x['process_id']==pid) for pid in sorted({x['process_id'] for x in rs})]
            row.update({stat+'_median':statistics.median(vals),stat+'_min':min(vals),stat+'_max':max(vals),stat+'_process_mean_min':min(means),stat+'_process_mean_max':max(means)})
        summary.append(row)
    write(root/'paired_raw.csv',paired);write(root/'summary.csv',summary)
    expected=len(m['cases'])*24*3*m['processes'];nchecks=len(m['cases'])*24*m['processes']+(len(m['cases'])*2 if m['smoke'] else 0)
    result=dict(valid=m['status']=='COMPLETED' and len(xs)==expected and len(cs)==nchecks and all(x['finite']=='True' and float(x['relative_error'])<=.002 for x in cs),rows=len(xs),expected_rows=expected,checks=len(cs),expected_checks=nchecks,worst_relative_error=max(float(x['relative_error']) for x in cs),source_sha=m['git_sha'],smoke=m['smoke'],exact_KV_buffers=True,noop_ratios={rep:dict(median=statistics.median(x['disabled_over_original'] for x in paired if x['representation']==rep),min=min(x['disabled_over_original'] for x in paired if x['representation']==rep),max=max(x['disabled_over_original'] for x in paired if x['representation']==rep)) for rep in ['INT4','FP16']})
    (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();print(json.dumps(analyze(a.run),indent=2))
