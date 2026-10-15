"""One unattended process: smoke then full per experiment; retain failed gates."""
import os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
for eid in ['E04b_end_to_end','E06_gqa']:
    cmd=[sys.executable,str(root/'kernel/tools/run_experiment.py'),'--experiment',eid]
    env=os.environ.copy();env['OFD_SMOKE']='1'
    smoke=subprocess.run(cmd,cwd=root,env=env).returncode
    print('SMOKE_STATUS',eid,smoke,flush=True)
    if smoke:
        print('FULL_BLOCKED_SMOKE_FAILURE',eid,flush=True)
        continue
    env.pop('OFD_SMOKE',None)
    full=subprocess.run(cmd,cwd=root,env=env).returncode
    print('FULL_STATUS',eid,full,flush=True)
