"""E10 online subgroup correctness, including the T+steps extra split."""
import argparse,csv,subprocess
from pathlib import Path
import torch
from ofd_kernel.runtime.online_cache import OnlineKV

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();rows=[]
    torch.manual_seed(1337);k=torch.randn(1,4,2048,256,device='cuda',dtype=torch.float16);v=torch.randn_like(k)
    q=[torch.randn(1,24,1,256,device='cuda',dtype=torch.float16) for _ in range(2)]
    kin=[torch.randn(1,4,1,256,device='cuda',dtype=torch.float16) for _ in range(2)];vin=[torch.randn_like(z) for z in kin]
    sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for rep in ['P','FP16']:
        refs=[]
        for g in [1,2,3,6]:
            store=OnlineKV(k,v,rep,2,group_q=g)
            for step in range(2):
                store.append(kin[step],vin[step]);out=store.attention(q[step]).clone()
                if g==1:refs.append(out)
                err=((out.float()-refs[step].float()).norm()/(refs[step].float().norm()+1e-12)).item();finite=bool(torch.isfinite(out).all().item())
                rows.append(dict(representation=rep,group_q=g,step=step,context=2048,capacity=2050,num_splits=3,relative_error=err,finite=finite,git_sha=sha))
                if not finite or err>.002:raise RuntimeError(str(rows[-1]))
            del store
    with a.out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('ONLINE_GROUPING_PASS',len(rows),'worst',max(x['relative_error'] for x in rows),flush=True)
if __name__=='__main__':main()
