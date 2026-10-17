"""Untimed E14 offline algebra; no production kernel changes."""
import csv, hashlib, json, math
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3]
CFG=json.loads((ROOT/'kernel/experiments/E14_operator_structure/execution.json').read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def tensor_hash(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def hadamard(d,device):
 h=torch.ones(1,1,dtype=torch.float64,device=device)
 while h.shape[0]<d:h=torch.cat([torch.cat([h,h],1),torch.cat([h,-h],1)],0)
 return h/math.sqrt(d)
def spectrum(x):
 s=torch.linalg.svdvals(x.double());p=s.square()/s.square().sum().clamp_min(1e-300)
 return dict(singular_values=s.tolist(),stable_rank=float(s.square().sum()/s[0].square().clamp_min(1e-300)),entropy_effective_rank=float(torch.exp(-(p*p.clamp_min(1e-300).log()).sum())),ranks=[dict(atol=a,rtol=r,rank=int((s>max(a,r*float(s[0]))).sum())) for a in CFG['rank_atol'] for r in CFG['rank_rtol']])
def norm(x):return float(torch.linalg.vector_norm(x))
def err(x,ref):return dict(abs_error=norm(x-ref),reference_norm=norm(ref),relative_error=norm(x-ref)/max(norm(ref),1e-30))
def center(z):return z-z.mean(-1,keepdim=True)
def csv_write(path,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for x in rows for k in x))
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for x in rows:w.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in x.items()})
def finite(x):
 if not bool(torch.isfinite(x).all()):raise RuntimeError('Non-finite E14 state/analysis')
def qdq(x):
 from ofd_kernel.reference.quant import quantize_int4_packed_groupwise,dequantize_int4_packed_groupwise
 o=quantize_int4_packed_groupwise(x,32)
 return dequantize_int4_packed_groupwise(o).double(),o
