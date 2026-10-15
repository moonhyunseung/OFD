"""Precondition validation only; does not mark E04b/E06 performance gates PASS."""
import contextlib,csv,json,math,subprocess,sys,traceback,uuid
from datetime import datetime,timezone
from pathlib import Path
import torch
from ofd_kernel.reference.fwht import fwht_normalized
from ofd_kernel.triton.fwht import fwht_into
from ofd_kernel.reference.attention import paged_quantized_variant_attention
from ofd_kernel.contracts import Variant
from ofd_kernel.runtime.launcher import TritonDecodeConfig,allocate_workspace,triton_paged_decode_core
from ofd_kernel.runtime.grouped_launcher import triton_paged_decode_grouped_core
from bench_gpu import to_cuda_cache,rel

ROOT=Path(__file__).resolve().parents[2]
class Tee:
    def __init__(self,*streams): self.streams=streams
    def write(self,s):
        for f in self.streams:f.write(s);f.flush()
    def flush(self):
        for f in self.streams:f.flush()

def checks(out):
    torch.manual_seed(1337)
    fw=[]
    for d in [64,128,256]:
        for rows in [1,24,256]:
            x=torch.randn(rows,d,device='cuda',dtype=torch.float16)
            y=torch.empty_like(x);z=torch.empty_like(x)
            fwht_into(x,y);fwht_into(y,z)
            ref=fwht_normalized(x)
            err=rel(ref,y);inv=rel(x,z)
            print('FWHT',d,rows,err,inv,flush=True)
            if max(err,inv)>0.002:raise RuntimeError('FWHT correctness failed')
            fw.append(dict(head_dim=d,rows=rows,relative_error=err,inverse_error=inv))
    with (out/'fwht_correctness.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fw[0]);w.writeheader();w.writerows(fw)
    rows=[]
    cfg=TritonDecodeConfig(split_size=64,tokens_per_block=16)
    for d in [64,128]:
        for r in [1,4,8]:
            b,hkv,t=3,2,257
            q=torch.randn(b,hkv*r,d,dtype=torch.float16)
            k=torch.randn(b,hkv,t,d,dtype=torch.float16)
            v=torch.randn_like(k)
            lens=torch.tensor([17,93,257])
            for variant in Variant:
                ref,cache=paged_quantized_variant_attention(q,k,v,variant,seq_lens=lens,group_size=32,page_size=16,split_size=64,seed=1337)
                kp,ks,vp,vs,bt,lg=to_cuda_cache(cache,torch.device('cuda'))
                qg=q.cuda();qw=torch.empty_like(qg);final=torch.empty_like(qg)
                if variant.rotate_k: fwht_into(qg,qw)
                else: qw.copy_(qg)
                ws=allocate_workspace(qg,lg,cfg,max_seq_len=t)
                wg=allocate_workspace(qg,lg,cfg,max_seq_len=t)
                triton_paged_decode_core(qw,kp,ks,vp,vs,bt,lg,config=cfg,workspace=ws,max_seq_len=t)
                triton_paged_decode_grouped_core(qw,kp,ks,vp,vs,bt,lg,config=cfg,workspace=wg,max_seq_len=t)
                group_err=rel(ws.out,wg.out)
                if variant.rotate_v: fwht_into(wg.out,final)
                else:final.copy_(wg.out)
                from ofd_kernel.runtime.gpu_ofd import gpu_ofd_decode
                qw.fill_(float('nan'))
                wrapped=gpu_ofd_decode(qg,kp,ks,vp,vs,bt,lg,variant=variant,config=cfg,workspace=ws,max_seq_len=t,q_work=qw,out_work=final)
                wrapper_err=rel(ref,wrapped.cpu())
                if not math.isfinite(wrapper_err) or wrapper_err>0.002:raise RuntimeError('Actual wrapper/oracle correctness failed')
                oracle_err=wrapper_err
                print('GQA',d,r,variant.value,group_err,oracle_err,flush=True)
                if not math.isfinite(oracle_err) or max(group_err,oracle_err)>0.002:
                    raise RuntimeError('Grouped GPU FWHT/oracle correctness failed')
                rows.append(dict(head_dim=d,q_per_kv=r,variant=variant.value,grouped_vs_qhead_error=group_err,grouped_gpu_fwht_vs_cpu_error=oracle_err))
                with (out/'grouped_correctness.csv').open('w') as f:
                    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    return dict(fwht_cases=len(fw),grouped_cases=len(rows),worst_fwht_error=max(float(x['relative_error']) for x in fw),worst_grouped_oracle_error=max(float(x['grouped_gpu_fwht_vs_cpu_error']) for x in rows))

def main():
    env=json.loads(subprocess.check_output([sys.executable,str(ROOT/'kernel/tools/collect_env.py'),'--root',str(ROOT)],text=True))
    if not env.get('claim_eligible'):raise RuntimeError('Clean Git commit required before GPU validation')
    out=ROOT/'kernel/results/implementation_validation'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    m=dict(schema_version=2,experiment_id='implementation_validation',environment=env,git_commit=env['git_commit'],git_dirty=env['git_dirty'],execution_status='RUNNING',claim_eligible_run=False,notes=['Implementation correctness only; E04b/E06 performance protocols remain blocked pending validation review.'],seed=1337,threshold=0.002)
    (out/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n')
    with (out/'run.log').open('w') as log,contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
        try:
            m['summary']=checks(out);m['execution_status']='COMPLETED';m['gate_status']='PASS'
        except Exception:
            m['execution_status']='FAILED';m['gate_status']='FAIL';traceback.print_exc()
        finally:
            m['artifacts']=sorted(p.name for p in out.iterdir() if p.name!='MANIFEST.json')
            (out/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n')
            print('RESULT',out,m['execution_status'])
    if m['execution_status']!='COMPLETED':raise RuntimeError('Implementation validation failed; see preserved log')
if __name__=='__main__':main()
