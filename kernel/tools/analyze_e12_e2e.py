"""Complete-model E12 no-empty-split control; paired wall and event effects."""
import argparse,csv,json,statistics,math
from pathlib import Path

def analyze(root):
    m=json.loads((root/'MANIFEST.json').read_text());rows=list(csv.DictReader((root/'e2e.csv').open()));expected=12 if m['smoke'] else 48;data={}
    for x in rows:
        rep=x['policy_label'].split('_')[0];key=(int(x['context']),int(x['batch']),rep,int(x['round']),x['execution'])
        assert key not in data and x['git_sha']==m['git_sha'] and x['status']=='OK'
        assert int(x['group_q'])==(2 if rep=='INT4' else 1)
        assert math.isfinite(float(x['decode_ms'])) and float(x['decode_ms'])>0
        assert int(x['capacity'])==int(x['context'])+(4 if m['smoke'] else 16)
        assert int(x['inactive_splits_first_step'])==0 and x['min_actual_splits']==x['num_splits']==x['max_actual_splits']
        data[key]=x
    summary=[]
    for t,b,rep in sorted({k[:3] for k in data}):
        a=[data[t,b,rep,r,'legacy'] for r in range(3)];z=[data[t,b,rep,r,'skip'] for r in range(3)]
        ratios=[float(x['decode_ms'])/float(y['decode_ms']) for x,y in zip(a,z)];gains=[float(x['decode_ms'])-float(y['decode_ms']) for x,y in zip(a,z)]
        summary.append(dict(context=t,batch=b,representation=rep,group_q=2 if rep=='INT4' else 1,legacy_ms_median=statistics.median(float(x['decode_ms']) for x in a),skip_ms_median=statistics.median(float(x['decode_ms']) for x in z),paired_speedup_median=statistics.median(ratios),paired_speedup_min=min(ratios),paired_speedup_max=max(ratios),paired_gain_ms_median=statistics.median(gains),legacy_event_ms_median=statistics.median(float(x['gpu_event_ms']) for x in a),skip_event_ms_median=statistics.median(float(x['gpu_event_ms']) for x in z),legacy_kv_bytes=a[0]['kv_bytes'],skip_kv_bytes=z[0]['kv_bytes'],inactive_splits_first_step=0,git_sha=m['git_sha']))
    with (root/'model_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    result=dict(valid=m['status']=='COMPLETED' and len(rows)==expected,rows=len(rows),expected_rows=expected,source_sha=m['git_sha'],smoke=m['smoke'],no_empty_splits=True,native_gdn_reference_fallback='falling back to its reference' in (root/'run.log').read_text(),groups=summary)
    (root/'model_analysis.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();r=analyze(a.run);print(json.dumps({k:v for k,v in r.items() if k!='groups'},indent=2))
