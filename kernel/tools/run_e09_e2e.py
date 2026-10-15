"""One process: committed source, online-cache correctness, smoke, full E2E."""
import json,os,subprocess,sys,uuid
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def run(smoke):
    env=json.loads(subprocess.check_output([sys.executable,'kernel/tools/collect_env.py','--root',str(ROOT)],cwd=ROOT,text=True))
    if not env.get('claim_eligible'):raise RuntimeError('Clean committed source required')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8];out=ROOT/'kernel/results/E09_model_decode_e2e'/stamp;out.mkdir(parents=True)
    cmds=([[sys.executable,'kernel/tools/validate_online_kv.py']] if smoke else [])+[[sys.executable,'kernel/tools/bench_e09_e2e.py','--out',str(out)]+(['--smoke'] if smoke else [])]
    m=dict(experiment_id='E09_model_decode_e2e',run_id=stamp,environment=env,git_commit=env['git_commit'],git_dirty=False,protocol=json.loads((ROOT/'kernel/experiments/E09_real_model/e2e_protocol.json').read_text()),commands=cmds,smoke=smoke,execution_status='RUNNING',claim_eligible_run=not smoke,scope='full_model_teacher_forced_online_decode',serving_claim=False)
    (out/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');e=os.environ.copy();e['PYTHONPATH']=str(ROOT/'kernel/src');e['TOKENIZERS_PARALLELISM']='false';rc=0
    with (out/'run.log').open('w') as log:
        for cmd in cmds:
            rc=subprocess.run(cmd,cwd=ROOT,env=e,stdout=log,stderr=subprocess.STDOUT).returncode
            if rc:break
    m.update(return_code=rc,execution_status='COMPLETED' if rc==0 else 'FAILED',artifacts=sorted(p.name for p in out.iterdir() if p.name!='MANIFEST.json'));(out/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');print('E09_E2E_RESULT',out,rc,flush=True);return rc
if __name__=='__main__':
    if run(True)==0:run(False)
