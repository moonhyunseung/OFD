"""E12 model control: frozen E09 decode, only inactive-split loop guard changes."""
import argparse,csv,gc,json,random,subprocess,sys,uuid
from pathlib import Path
from datetime import datetime,timezone
import torch
from bench_e09_e2e import Sweep,C,ROOT
from bench_e10_grouping import error,write_csv
from ofd_kernel.runtime import online_cache as oc
from ofd_kernel.runtime.guarded_launcher import guarded_paged_decode_core
ORIGINAL=oc.OnlineKV.attention
MODE='legacy'

def attention(self,q):
    if MODE=='legacy':return ORIGINAL(self,q)
    self.q.copy_(q[:,:,0,:]);lens=self.lens[self.pos-self.start-1]
    out=guarded_paged_decode_core(self.q,self.kp,self.ks,self.vp,self.vs,self.bt,lens,config=oc.CFG,workspace=self.ws,max_seq_len=self.capacity,representation='FP16' if self.variant=='FP16' else 'INT4',group_q=self.group_q,skip_empty=True)
    return out[:,None,:,:]
oc.OnlineKV.attention=attention

def online_gate(out):
    global MODE
    torch.manual_seed(C['seed']);checks=[];sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    k=torch.randn(1,4,2048,256,device='cuda',dtype=torch.float16);v=torch.randn_like(k)
    for rep in ['P','FP16']:
        for g in [1,2,3,6]:
            a=oc.OnlineKV(k,v,rep,2,group_q=g);b=oc.OnlineKV(k,v,rep,2,group_q=g)
            for step in range(2):
                ki=torch.randn(1,4,1,256,device='cuda',dtype=torch.float16);vi=torch.randn_like(ki);q=torch.randn(1,24,1,256,device='cuda',dtype=torch.float16)
                a.append(ki,vi);b.append(ki,vi);ref=ORIGINAL(a,q);MODE='skip';y=b.attention(q);e=error(ref,y);finite=bool(torch.isfinite(y).all().item())
                checks.append(dict(representation=rep,group_q=g,step=step,capacity=2050,num_splits=3,relative_error=e,finite=finite,git_sha=sha));write_csv(out/'correctness_online.csv',checks)
                if not finite or e>.002:raise RuntimeError('Online guard equivalence failed')
            del a,b
    MODE='legacy';torch.cuda.empty_cache()

class Confirmation(Sweep):
    def write(self,row):
        t=row['context'];cap=t+self.steps;ns=(cap+1023)//1024
        row.update(execution=MODE,group_q=self.group_q,capacity=cap,num_splits=ns,min_actual_splits=(t+1+1023)//1024,max_actual_splits=ns,inactive_splits_first_step=ns-(t+1+1023)//1024,git_sha=self.sha)
        super().write(row)
    def matrix(self):
        global MODE
        cfg=json.loads((ROOT/'kernel/experiments/E12_skip_empty/protocol.json').read_text())
        for t,b in ([[128,1]] if self.smoke else cfg['fullmodel']['shapes']):
            self.template('prose','evaluation',t,b)
            choices=[('FP16_legacy','FP16',1,'legacy'),('FP16_skip','FP16',1,'skip'),('INT4_legacy','P',2,'legacy'),('INT4_skip','P',2,'skip')]
            for rnd in range(3):
                order=choices.copy();random.Random(C['seed']+t+b*17+rnd).shuffle(order)
                for label,rep,g,mode in order:
                    self.group_q=g;MODE=mode;self.run({str(i):rep for i in C['layers']},label,rnd,phase='E12_confirmation')
            self.refs.clear()
        self.f.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--out',type=Path);a=p.parse_args()
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    gate=json.loads((ROOT/'kernel/experiments/E12_skip_empty/VALIDATED_GATE.json').read_text())
    if not gate['lowlevel_valid']:raise RuntimeError('Lowlevel gate closed')
    if a.worker:
        if a.smoke:online_gate(a.out)
        z=Confirmation(a.out,a.smoke);z.sha=sha;z.group_q=1;z.matrix();return
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise RuntimeError('Clean source required')
    out=ROOT/'kernel/results/E10_gqa_grouping'/('E12_model_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);out.mkdir()
    env=json.loads(subprocess.check_output([sys.executable,'kernel/tools/collect_env.py','--root',str(ROOT)],text=True))
    m=dict(experiment='E12_complete_model_guard_control',git_sha=sha,smoke=a.smoke,status='RUNNING',gate=gate,environment=env,original_e09_protocol=C,protocol=json.loads((ROOT/'kernel/experiments/E12_skip_empty/protocol.json').read_text()),opportunity='Uniform online append makes all allocated splits active for these T divisible by1024; guard is a no-op overhead control, not a padded serving speedup test',reference='Speed-only; finite logits and separate online equivalence; no native logits reference')
    (out/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n')
    with (out/'run.log').open('w') as f:
        rc=subprocess.run([sys.executable,__file__,'--worker','--out',str(out)]+(['--smoke'] if a.smoke else []),stdout=f,stderr=subprocess.STDOUT,cwd=ROOT).returncode
    m.update(status='COMPLETED' if rc==0 else 'FAILED',return_code=rc);(out/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');print('E12_MODEL',out,rc,flush=True)
    if rc:raise RuntimeError('Model confirmation failed; logs retained')
if __name__=='__main__':main()
