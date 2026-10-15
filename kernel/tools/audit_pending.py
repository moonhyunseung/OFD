"""Persist observed readiness; never turn missing hardware/model gates into PASS."""
import importlib.util,json,subprocess,sys,uuid
from datetime import datetime,timezone
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
env=json.loads(subprocess.check_output([sys.executable,str(ROOT/'kernel/tools/collect_env.py'),'--root',str(ROOT)],text=True))
if not env.get('claim_eligible'):raise RuntimeError('Clean source commit required')
cc=torch.cuda.get_device_capability(0);gpu=torch.cuda.get_device_name(0)
for eid,status,reason in [
 ('E08_blackwell_tuning','READY_UNMEASURED' if cc[0] in (10,12) else 'BLOCKED_ENV',f'Observed GPU {gpu}, compute capability {cc}. E08 requires a Blackwell execution target and a profiler-backed tuning run; H200 tuning cannot be labeled Blackwell tuning.'),
 ('E09_real_model','BLOCKED_PROTOCOL','E08 has no completed Blackwell run. Real-model serving integration, pinned model revision, held-out calibration/evaluation data and policy-validation contract are not yet implemented/frozen.')]:
    cfg=json.loads((ROOT/'kernel/experiments'/eid/'config.json').read_text())
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=ROOT/'kernel/results'/eid/stamp;out.mkdir(parents=True)
    m=dict(schema_version=2,experiment_id=eid,run_id=stamp,git_commit=env['git_commit'],git_dirty=env['git_dirty'],environment=env,config=cfg,execution_status=status,gate_status='PENDING',claim_eligible_run=False,return_code=2,artifacts=['run.log'],notes=[reason],gpu=gpu,compute_capability=cc,transformers_importable=importlib.util.find_spec('transformers') is not None,commands=[[sys.executable,str(Path(__file__).resolve())]])
    (out/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');(out/'run.log').write_text(reason+'\n')
    print(eid,status,out)
