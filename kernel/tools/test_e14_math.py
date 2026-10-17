"""Independent algebra/bound and FP16-edge gates for E14; no model required."""
import argparse,json,subprocess
from pathlib import Path
import torch
from e14.common import ROOT,hadamard,qdq,norm,center,spectrum
from analyze_e14 import planes,truncated_planes,analyze_group

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args();torch.manual_seed(1337);torch.set_num_threads(4);dev=a.device;checks=[]
 def check(name,x,limit):
  val=float(x);checks.append(dict(check=name,value=val,limit=limit,passed=val<=limit));assert val<=limit,(name,val)
 check('zero_spectrum_effective_rank',spectrum(torch.zeros(6,256,device=dev))['entropy_effective_rank'],0)
 q=torch.randn(6,256,device=dev,dtype=torch.float64);k=torch.randn(17,256,device=dev,dtype=torch.float64);v=torch.randn_like(k);h=hadamard(256,dev)
 check('orthogonal_score_symmetry',(q@k.T-(q@h)@(k@h).T).abs().max(),1e-10)
 b=torch.kron(torch.eye(8,device=dev,dtype=torch.float64),hadamard(32,dev))
 check('H32_each_block_energy_invariant',(q.reshape(6,8,32).square().sum(-1)-(q@b).reshape(6,8,32).square().sum(-1)).abs().max(),1e-10)
 z=q@k.T/16;pp=z.softmax(-1)
 check('softmax_constant_quotient',(pp-center(z).softmax(-1)).abs().max(),1e-12)
 kh,ko=qdq(k);vh,vo=qdq(v)
 def codes(o):return torch.stack([o.packed&15,o.packed>>4],-1).reshape(17,8,32).double()
 ku=codes(ko);vu=codes(vo);zz=torch.zeros_like(z);yy=torch.zeros(6,256,device=dev,dtype=torch.float64)
 for g in range(8):
  qg=q[:,g*32:(g+1)*32];sk=ko.scale[:,g].double();sv=vo.scale[:,g].double()
  zz+=(qg@ku[:,g].T)*sk-8*qg.sum(-1,keepdim=True)*sk
  pv=pp*sv;yy[:,g*32:(g+1)*32]=pv@vu[:,g]-8*pv.sum(-1,keepdim=True)
 check('unsigned_K_rank1_scale_identity',(zz-q@kh.T).abs().max(),1e-9)
 check('unsigned_V_rank1_scale_identity',(yy-pp@vh).abs().max(),1e-10)
 special=torch.tensor([0.,-0.,2**-24,-2**-24,2**-14,65504.,-65504.,1.],device=dev,dtype=torch.float64).repeat(4).reshape(1,32)
 full,ints,info=planes(special);check('FP16_extreme_exact_planes',(truncated_planes(ints,info['nonempty_planes_max'])-full).abs().max(),0)
 for subset in [[],[0],[0,1,2],list(range(17))]:
  keep=torch.ones(17,device=dev,dtype=torch.bool);keep[subset]=False;delta=pp[:,~keep].sum(-1);er=(pp[:,~keep]@v[~keep]).norm(dim=-1);bound=delta*v.norm(dim=-1).max();check('unrenormalized_V_support_bound_'+str(len(subset)),(er-bound).clamp_min(0).max(),1e-10)
 # End-to-end analyzer fixture, with all metrics and disjoint past/future steps.
 qs=torch.randn(8,6,256,device=dev,dtype=torch.float64);nk=torch.randn(8,256,device=dev,dtype=torch.float64);nv=torch.randn_like(nk);tables={};bases=[]
 analyze_group(qs,k,v,nk,nv,None,1/16,dict(domain='synthetic',partition='test',context=17,layer=-1,kv_head=0),tables,bases)
 check('fixture_block_rows',abs(len(tables['blocks'])-8*5*2*8),0)
 check('fixture_quantization_rows',abs(len(tables['quantization'])-8*5),0)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps({'status':'PASS','git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'checks':checks,'fixture_counts':{k:len(v) for k,v in tables.items()}},indent=2));print('E14_MATH_PASS',len(checks),flush=True)
if __name__=='__main__':main()
