"""E10 confirmation only: reuse E09 complete-model teacher-forced decode."""
import argparse,gc,json,os,random,subprocess,sys,traceback,uuid
from datetime import datetime,timezone
from pathlib import Path
import torch
from bench_e09_e2e import Sweep,C,ROOT

class Confirmation(Sweep):
    def __init__(self,out,smoke,policy):
        self.policy=policy;self.group_q=1
        super().__init__(out,smoke)
    def write(self,row):
        rep='FP16' if row['policy_label'].startswith('FP16') else 'INT4'
        row.update(group_q=self.group_q,representation=rep,capacity=row['context']+self.steps,num_splits=(row['context']+self.steps+1023)//1024,stage1_program_count=row['batch']*4*(6//self.group_q)*((row['context']+self.steps+1023)//1024),W1=row['batch']*24*((row['context']+self.steps+1023)//1024),git_sha=self.policy['confirmation_source_sha'],group_policy_source_sha=self.policy['lowlevel_source_sha'])
        super().write(row)
    def matrix(self):
        shapes=[[128,1]] if self.smoke else self.policy['fullmodel_shapes']
        for t,b in shapes:
            try:self.template('prose','evaluation',t,b)
            except (torch.OutOfMemoryError,MemoryError) as e:
                (self.out/f'OOM_setup_{t}_{b}.json').write_text(json.dumps(dict(context=t,batch=b,status='HOST_TEMPLATE_LIMIT' if isinstance(e,MemoryError) else 'PREFILL_SETUP_OOM',reason=str(e)))+'\n');gc.collect();torch.cuda.empty_cache();continue
            choices=[('FP16_g1','FP16',1),('FP16_gbest','FP16',self.policy['fixed']['FP16']),('INT4_g1','P',1),('INT4_gbest','P',self.policy['fixed']['INT4'])]
            for rnd in range(3):
                order=choices.copy();random.Random(C['seed']+t+b*17+rnd).shuffle(order)
                for label,variant,g in order:
                    self.group_q=g
                    try:self.run({str(i):variant for i in C['layers']},label,rnd,phase='confirmation')
                    except torch.OutOfMemoryError as e:
                        (self.out/f'OOM_{t}_{b}_{label}_{rnd}.json').write_text(json.dumps(dict(context=t,batch=b,label=label,round=rnd,status='DECODE_SETUP_OR_EXECUTION_OOM',reason=str(e)))+'\n');self.active=None;gc.collect();torch.cuda.empty_cache()
            self.refs.clear()
        self.f.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--out',type=Path);a=p.parse_args()
    policy_path=ROOT/'kernel/experiments/E10_gqa_grouping/frozen_policy.json';policy=json.loads(policy_path.read_text())
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();policy['confirmation_source_sha']=sha
    if a.worker:
        Confirmation(a.out,a.smoke,policy).matrix();return
    if not policy['lowlevel_valid']:raise RuntimeError('Low-level validation gate is closed')
    env=json.loads(subprocess.check_output([sys.executable,'kernel/tools/collect_env.py','--root',str(ROOT)],cwd=ROOT,text=True))
    if not env.get('claim_eligible'):raise RuntimeError('Clean source required')
    out=ROOT/'kernel/results/E10_gqa_grouping'/('confirmation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True)
    m=dict(experiment='E10_model_confirmation',environment=env,git_sha=sha,smoke=a.smoke,status='RUNNING',claim_eligible=not a.smoke,policy=policy,original_e09_protocol=C,timing='unchanged E09 full model decode; all64layers/onlineappend/LMhead; prefill and sampling excluded',capacity_definition='T+decode_steps; Ns=ceil((T+decode_steps)/1024)',reference_note='No native logits reference collected; low-level numerical equivalence and finite model outputs checked; speed confirmation only')
    (out/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--out',str(out)]+(['--smoke'] if a.smoke else [])
    with (out/'run.log').open('w') as log:
        rc=0
        if a.smoke:rc=subprocess.run([sys.executable,'kernel/tools/validate_e10_online.py','--out',str(out/'correctness_online.csv')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
        if rc==0:rc=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
    m.update(status='COMPLETED' if rc==0 else 'FAILED',return_code=rc,commands=[cmd]);(out/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');print('E10_CONFIRMATION',out,rc,flush=True)
    if rc:raise RuntimeError('Confirmation failed, logs retained')
if __name__=='__main__':main()
