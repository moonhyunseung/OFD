"""Paired effects for frozen E11 interventions; no significance cutoffs."""
import csv,json,statistics,hashlib,argparse,math
from pathlib import Path

def write(p,rs):
    if not rs:return
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)

def analyze(root):
    m=json.loads((root/'MANIFEST.json').read_text());raw=list(csv.DictReader((root/'raw.csv').open()));checks=list(csv.DictReader((root/'correctness.csv').open()))
    axes=['family','T','B','Hkv','R','alias','cap'];data={};sums={}
    for x in raw:
        k=tuple(x[a] for a in axes)+(x['representation'],x['group_q'],x['process_id'],x['round_id'])
        if k in data:raise RuntimeError('Duplicate')
        assert x['git_sha']==m['git_sha'];assert math.isfinite(float(x['latency_ms'])) and float(x['latency_ms'])>0
        data[k]=float(x['latency_ms']);sums.setdefault(k[:-2],[]).append(float(x['latency_ms']))
        ns=(int(x['capacity'])+1023)//1024
        assert int(x['stage1_program_count'])==int(x['B'])*int(x['Hkv'])*(int(x['R'])//int(x['group_q']))*ns
    identities={}
    for x in raw:
        kk=(x['T'],x['B'],x['Hkv'],x['representation'],x['process_id'])
        identities.setdefault(kk,set()).add(x['kv_buffer_identity'])
    assert all(len(v)==1 for v in identities.values()),'KV buffers changed within controlled axis'
    summary=[dict(zip(axes+['representation','group_q'],k),n=len(v),median_ms=statistics.median(v),min_ms=min(v),max_ms=max(v)) for k,v in sums.items()]
    write(root/'summary.csv',summary)
    effects=[];contrasts=[]
    for k,v in data.items():
        family,t,b,h,r,alias,cap,rep,g,pid,rnd=k
        if rep=='INT4' and g!='1':
            i1=k[:8]+('1',pid,rnd);fg=k[:7]+('FP16',g,pid,rnd);f1=k[:7]+('FP16','1',pid,rnd)
            if all(z in data for z in [i1,fg,f1]):
                gi=data[i1]-v;gf=data[f1]-data[fg];effects.append(dict(zip(axes,k[:7]),group_q=g,process_id=pid,round_id=rnd,GI_ms=gi,GF_ms=gf,D_ms=gi-gf))
        base=list(k)
        if family=='inactive_splits':base[6]='1'
        elif family=='physical_reuse':base[5]='1'
        elif family=='batch_head_repartition':base[2]='1';base[3]='4' if m['smoke'] else '16'
        elif family=='q_repeat':
            if g!='1':continue
            base[4]='1'
        ref=data.get(tuple(base))
        if ref is not None:contrasts.append(dict(zip(axes,k[:7]),representation=rep,group_q=g,process_id=pid,round_id=rnd,latency_ratio=v/ref,delta_ms=v-ref,per_query_head_ratio=(v/ref)/(int(r) if family=='q_repeat' else 1)))
    write(root/'effects_raw.csv',effects);write(root/'contrasts_raw.csv',contrasts)
    esum=[]
    for key in sorted({tuple(x[a] for a in axes+['group_q']) for x in effects}):
        rs=[x for x in effects if tuple(x[a] for a in axes+['group_q'])==key];row=dict(zip(axes+['group_q'],key),n=len(rs))
        for stat in ['GI_ms','GF_ms','D_ms']:
            vals=[x[stat] for x in rs];means=[statistics.mean(x[stat] for x in rs if x['process_id']==p) for p in sorted({x['process_id'] for x in rs})]
            row.update({stat+'_median':statistics.median(vals),stat+'_min':min(vals),stat+'_max':max(vals),stat+'_process_mean_min':min(means),stat+'_process_mean_max':max(means)})
        esum.append(row)
    write(root/'effects_summary.csv',esum)
    ed={tuple(x[a] for a in axes+['group_q','process_id','round_id']):x for x in effects};interactions=[]
    for key,x in ed.items():
        base=list(key);family=key[0]
        if family=='inactive_splits':base[6]='1'
        elif family=='physical_reuse':base[5]='1'
        elif family=='batch_head_repartition':base[2]='1';base[3]='4' if m['smoke'] else '16'
        else:base[4]=x['group_q']
        ref=ed.get(tuple(base))
        if ref is not None:interactions.append(dict(x,delta_D_ms=x['D_ms']-ref['D_ms'],delta_GI_ms=x['GI_ms']-ref['GI_ms'],delta_GF_ms=x['GF_ms']-ref['GF_ms']))
    write(root/'intervention_interactions_raw.csv',interactions)
    csum=[]
    for key in sorted({tuple(x[a] for a in axes+['representation','group_q']) for x in contrasts}):
        rs=[x for x in contrasts if tuple(x[a] for a in axes+['representation','group_q'])==key]
        vals=[x['latency_ratio'] for x in rs];means=[statistics.mean(x['latency_ratio'] for x in rs if x['process_id']==p) for p in sorted({x['process_id'] for x in rs})]
        csum.append(dict(zip(axes+['representation','group_q'],key),n=len(rs),ratio_median=statistics.median(vals),ratio_min=min(vals),ratio_max=max(vals),process_mean_min=min(means),process_mean_max=max(means)))
    write(root/'contrasts_summary.csv',csum)
    # Secondary cross-family diagnostic: same launched programs and unique-address bytes,
    # but 4x active token traversal. Allocation size and pointer identity differ across T.
    triads=[]
    if not m['smoke']:
        lookup={tuple(x[a] for a in axes+['representation','group_q','process_id','round_id']):x for x in raw}
        for b in ['1','4']:
            for rep in ['INT4','FP16']:
                for g in ['1','2','3','6']:
                    for pid in ['0','1','2']:
                        for rnd in ['0','1','2']:
                            suffix=(rep,g,pid,rnd)
                            ka=('inactive_splits','8192',b,'4','6','1','4')+suffix
                            kb=('physical_reuse','32768',b,'4','6','4','1')+suffix
                            kc=('physical_reuse','32768',b,'4','6','1','1')+suffix
                            if not all(k in lookup for k in [ka,kb,kc]):continue
                            a,z,c=[lookup[k] for k in [ka,kb,kc]]
                            assert a['stage1_program_count']==z['stage1_program_count']==c['stage1_program_count']
                            assert a['kv_unique_address_bytes']==z['kv_unique_address_bytes']
                            triads.append(dict(batch=b,representation=rep,group_q=g,process_id=pid,round_id=rnd,active_traversal_ratio=float(z['latency_ms'])/float(a['latency_ms']),unique_footprint_ratio=float(c['latency_ms'])/float(z['latency_ms']),program_count=a['stage1_program_count'],A_ms=a['latency_ms'],B_ms=z['latency_ms'],C_ms=c['latency_ms']))
    write(root/'secondary_equal_program_footprint_raw.csv',triads)
    expected=sum(2*sum(c['R']%g==0 for g in [1,2,3,6]) for c in m['cases'])*3*m['independent_processes']
    valid=m['status']=='COMPLETED' and len(raw)==expected and len(checks)==expected//3 and all(x['finite']=='True' and float(x['relative_error'])<=.002 for x in checks)
    result=dict(valid=valid,rows=len(raw),expected_rows=expected,checks=len(checks),worst_relative_error=max(float(x['relative_error']) for x in checks),source_sha=m['git_sha'],smoke=m['smoke'])
    (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();print(json.dumps(analyze(a.run),indent=2))
