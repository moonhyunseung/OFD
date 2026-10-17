"""Native Qwen state capture for E14. Untimed; reuses E09 model/data/prefill."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse, copy, gc, json, subprocess, sys, uuid, traceback
from pathlib import Path
from datetime import datetime,timezone
import torch
from e14.common import CFG,ROOT,sha

def capture(out,smoke):
 import bench_e09_e2e as e09
 from transformers import AttentionInterface
 from transformers.integrations.sdpa_attention import sdpa_attention_forward
 cfg={**CFG,**(CFG['smoke'] if smoke else {})}
 if 'H200' not in torch.cuda.get_device_name():raise RuntimeError('E14 target must be H200')
 torch.manual_seed(CFG['seed']);torch.cuda.manual_seed_all(CFG['seed'])
 torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 e09.C={**e09.C,'steps':cfg['decode_steps'],'warmup_steps':0}
 z=e09.Sweep(out,smoke);z.steps=cfg['decode_steps'];z.warm=0
 active={};weights={};records=[]
 def gate_hook(li):
  def hook(module,args,output):
   if active and output.shape[1]==1:
    gate=output.reshape(1,1,24,512).chunk(2,-1)[1].sigmoid()
    active.setdefault('gates',{})[li]=gate.reshape(4,6,256)[cfg['kv_heads']].detach().cpu()
  return hook
 handles=[mod.q_proj.register_forward_hook(gate_hook(mod.layer_idx)) for mod in z.model.modules() if hasattr(mod,'layer_idx') and mod.layer_idx in cfg['layers'] and hasattr(mod,'q_proj') and hasattr(mod,'o_proj')]
 assert len(handles)==len(cfg['layers'])
 def backend(module,q,k,v,mask,dropout=0.,scaling=None,**kw):
  if q.shape[2]>1:
   bias=None if q.shape[2]==k.shape[2] else e09.causal_lower_right(q.shape[2],k.shape[2])
   y=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=bias,is_causal=bias is None,enable_gqa=True,scale=scaling)
   return y.transpose(1,2),None
  if active and module.layer_idx in cfg['layers']:
   li=module.layer_idx;assert q.shape==(1,24,1,256) and k.shape[1]==4 and k.shape[-1]==256
   state=active['states'].setdefault(li,{'q':[],'new_k':[],'new_v':[],'gate':[]})
   heads=cfg['kv_heads']
   if not state['q']:
    state['base_k']=k[0,heads,:-1,:].detach().cpu();state['base_v']=v[0,heads,:-1,:].detach().cpu()
    state['scaling']=float(scaling if scaling is not None else 256**-.5)
   state['q'].append(q[0,:,0,:].reshape(4,6,256)[heads].detach().cpu())
   state['gate'].append(active['gates'][li])
   state['new_k'].append(k[0,heads,-1,:].detach().cpu());state['new_v'].append(v[0,heads,-1,:].detach().cpu())
   if li not in weights:
    w=module.o_proj.weight.detach().cpu();torch.save(w,out/f'wo_layer{li}.pt')
    weights[li]={'file':f'wo_layer{li}.pt','sha256':sha(out/f'wo_layer{li}.pt'),'shape':list(w.shape),'module_class':type(module).__name__,'q_proj_out_features':getattr(module.q_proj,'out_features',None),'WO_scope':'captured native sigmoid gate held fixed, then original WO; KV-group contribution only'}
  return sdpa_attention_forward(module,q,k,v,mask,dropout=dropout,scaling=scaling,**kw)
 AttentionInterface.register('ofd_online',backend)
 try:
  for domain in cfg['domains']:
   for t in cfg['contexts']:
    z.template(domain,CFG['partition'],t,1)
    cache=e09.move_cache(copy.deepcopy(z.base),'cuda');active.update(states={})
    with torch.inference_mode():
     for step in range(cfg['decode_steps']):
      y=z.model(input_ids=z.inputs[step],position_ids=z.positions[step],attention_mask=z.mask,past_key_values=cache,use_cache=True,logits_to_keep=1)
      if not bool(torch.isfinite(y.logits).all()):raise RuntimeError('Nonfinite logits')
      cache=y.past_key_values;del y
    assert set(active['states'])==set(cfg['layers'])
    for li,state in active['states'].items():
     for name in ['q','new_k','new_v','gate']:state[name]=torch.stack(state[name]);assert state[name].shape[0]==cfg['decode_steps']
     name=f'{domain}_T{t}_L{li}.pt';torch.save(state,out/name)
     records.append(dict(file=name,sha256=sha(out/name),domain=domain,partition=CFG['partition'],context=t,layer=li,kv_heads=cfg['kv_heads'],steps=cfg['decode_steps'],scaling=state['scaling']))
    (out/'capture_index.json').write_text(json.dumps({'states':records,'weights':weights,'protocol':cfg},indent=2))
    print('CAPTURE',domain,t,len(active['states']),flush=True)
    active.clear();del cache;gc.collect();torch.cuda.empty_cache()
 finally:
  z.f.close()
  for handle in handles:handle.remove()
 # Empty E09 writer is not an E14 timing result.
 (out/'e2e.csv').unlink(missing_ok=True)
 if not smoke:
  from huggingface_hub import snapshot_download
  snapshot=Path(snapshot_download(CFG['model_id'],revision=CFG['model_revision'],local_files_only=True))
  shards=sorted(snapshot.glob('*.safetensors'));assert shards
  (out/'checkpoint_hashes.json').write_text(json.dumps({p.name:sha(p) for p in shards},indent=2))

def main():
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--out',type=Path);p.add_argument('--analyze',action='store_true');a=p.parse_args()
 if a.worker:capture(a.out,a.smoke);return
 if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise RuntimeError('Clean source required')
 out=a.out or ROOT/'kernel/results/E14_operator_structure'/('smoke_' if a.smoke else 'capture_')/ (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True,exist_ok=False)
 m={'experiment':'E14','execution_status':'RUNNING','scientific_gate':'SMOKE_ONLY' if a.smoke else 'OBSERVATIONAL_NO_E15_ACCEPTANCE','smoke':a.smoke,'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'execution_protocol':CFG,'environment':json.loads(subprocess.check_output([sys.executable,'kernel/tools/collect_env.py','--root',str(ROOT)],cwd=ROOT,text=True)),'capture_contract':'post-RoPE Q/K, native FP16 all-token K/V; FP64 offline operator diagnostics; no speed/quality claim'}
 (out/'MANIFEST.json').write_text(json.dumps(m,indent=2))
 with (out/'run.log').open('w') as f:rc=subprocess.run([sys.executable,__file__,'--worker','--out',str(out)]+(['--smoke'] if a.smoke else []),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
 m['execution_status']='CAPTURED' if rc==0 else 'FAILED';m['return_code']=rc;(out/'MANIFEST.json').write_text(json.dumps(m,indent=2));print('E14_CAPTURE',out,rc,flush=True)
 if rc:raise RuntimeError('Capture failed; raw partials and log retained')
 if a.analyze:
  with (out/'analysis.log').open('w') as f:rc=subprocess.run([sys.executable,'kernel/tools/analyze_e14.py',str(out),'--device','cuda'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
  
  if rc==0:
   with (out/'summary.log').open('w') as f:rc=subprocess.run([sys.executable,'kernel/tools/summarize_e14.py',str(out)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
  m['analysis_return_code']=rc;m['execution_status']='COMPLETED' if rc==0 else 'ANALYSIS_FAILED';(out/'MANIFEST.json').write_text(json.dumps(m,indent=2))
  if rc:raise RuntimeError('Analysis failed; capture reusable')
if __name__=='__main__':main()
