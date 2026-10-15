"""Unattended smoke then full layer replay, preserving independent manifests."""
import json,os,subprocess,sys,uuid
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def run(smoke):
    env=json.loads(subprocess.check_output([sys.executable,'kernel/tools/collect_env.py','--root',str(ROOT)],cwd=ROOT,text=True))
    if not env.get('claim_eligible'):raise RuntimeError('Clean committed source required')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=ROOT/'kernel/results/E09_decode_layers'/stamp;out.mkdir(parents=True)
    cmd=[sys.executable,'kernel/tools/bench_e09_layers.py','--out',str(out)]+(['--smoke'] if smoke else [])
    m=dict(experiment_id='E09_decode_layers',run_id=stamp,git_commit=env['git_commit'],git_dirty=False,environment=env,protocol=json.loads((ROOT/'kernel/experiments/E09_real_model/decode_protocol.json').read_text()),commands=[cmd],execution_status='RUNNING',claim_eligible_run=False,scope='decode_layer_replay_not_serving_e2e',smoke=smoke)
    (out/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n')
    e=os.environ.copy();e['PYTHONPATH']=str(ROOT/'kernel/src');e['TOKENIZERS_PARALLELISM']='false'
    with (out/'run.log').open('w') as log: rc=subprocess.run(cmd,cwd=ROOT,env=e,stdout=log,stderr=subprocess.STDOUT).returncode
    m.update(return_code=rc,execution_status='COMPLETED' if rc==0 else 'FAILED',artifacts=sorted(p.name for p in out.iterdir() if p.name!='MANIFEST.json'))
    (out/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');print('E09_REPLAY_RESULT',out,rc,flush=True)
    return rc
if __name__=='__main__':
    if run(True)==0:run(False)
