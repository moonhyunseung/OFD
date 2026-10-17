"""E14 FP64 structural diagnostics. All-token softmax, explicit finite-Q radix model."""
import argparse,json,math,subprocess
from pathlib import Path
import torch
from e14.common import CFG,ROOT,sha,tensor_hash,hadamard,spectrum,norm,err,center,csv_write,finite,qdq

def planes(x):
 """Exact FP16 signed-magnitude binary planes, lsb fixed at 2^-24."""
 h=x.half();finite(h);integers=(h.double()*2**24).to(torch.int64)
 # Largest finite FP16 fits in 40 magnitude bits. No int64 overflow.
 mag=integers.abs().reshape(-1,32);used=torch.stack([((mag>>i)&1).any(-1) for i in range(40)],-1)
 pos=torch.arange(40,device=x.device)
 lo=torch.where(used,pos,40).amin(-1);hi=torch.where(used,pos,-1).amax(-1)
 count=used.sum(-1);span=torch.where(count>0,hi-lo+1,0)
 assert torch.equal(integers.double()*2**-24,h.double())
 return h.double(),integers,dict(nonempty_planes_mean=float(count.double().mean()),nonempty_planes_max=int(count.max()),contiguous_span_mean=float(span.double().mean()),contiguous_span_max=int(span.max()),fp16_rounding_error=err(h.double(),x))

def truncated_planes(integers,n):
 m=integers.abs().reshape(-1,32);used=torch.stack([((m>>i)&1).any(-1) for i in range(39,-1,-1)],-1)
 keep=used&(used.to(torch.int64).cumsum(-1)<=n);v=torch.zeros_like(m)
 for j,i in enumerate(range(39,-1,-1)):v+=(((m>>i)&1)<<i)*keep[:,j,None]
 return (v.reshape_as(integers)*integers.sign()).double()*2**-24

def analyze_group(qs,k0,v0,nk,nv,wo,scale,meta,tables,basis_records,gates=None):
 device=qs.device;d=qs.shape[-1];r=qs.shape[-2];ident=torch.eye(d,device=device,dtype=torch.float64)
 train=qs[:CFG['basis_train_steps']].reshape(-1,d)
 _,_,vt=torch.linalg.svd(train,full_matrices=True);past=vt.T
 fixed={'identity':ident,'block_H32':torch.kron(torch.eye(8,device=device,dtype=torch.float64),hadamard(32,device)),'full_H256':hadamard(256,device),'past_window_svd':past}
 def emit(table,**x):tables.setdefault(table,[]).append({**meta,**x})
 for name,a in fixed.items():basis_records.append({**meta,'basis':name,'definition':'normalized Sylvester' if 'H' in name else 'identity' if name=='identity' else 'right singular vectors of first four query steps, frozen; future only deployment diagnostic','sha256':tensor_hash(a),'orthogonality_max_abs':float((a.T@a-ident).abs().max())})
 # Nonoverlapping windows, full spectra and all declared rank tolerances.
 for w in CFG['window_sizes']:
  for start in range(0,len(qs)-w+1,w):emit('rank',kind='window',window=w,start=start,**spectrum(qs[start:start+w].reshape(-1,d)))
 for lag in CFG['drift_lags']:
  for step in range(lag,len(qs)):
   _,_,a=torch.linalg.svd(qs[step-lag],full_matrices=False);_,_,b=torch.linalg.svd(qs[step],full_matrices=False)
   for m in CFG['drift_subspace_dimensions']:
    sv=torch.linalg.svdvals(a[:m]@b[:m].T).clamp(0,1)
    emit('drift',step=step,lag=lag,subspace_dimension=m,angles_rad=torch.acos(sv).tolist(),chordal_squared=float((1-sv.square()).sum()))
 k=k0;v=v0
 for step,q in enumerate(qs):
  k=torch.cat([k,nk[step:step+1]],0);v=torch.cat([v,nv[step:step+1]],0);finite(q);finite(k);finite(v)
  z=(q@k.T)*scale;p=z.softmax(-1);y=p@v
  gate=torch.ones_like(y) if gates is None else gates[step]
  ywo=torch.einsum('rd,rdh->h',y*gate,wo) if wo is not None else None
  denom=max(norm(center(z)),1e-30)
  def output_stats(yp):
   o=err(yp,y);o['per_qhead']=[err(yp[i],y[i]) for i in range(r)]
   if wo is not None:o['WO_linear_group_contribution']=err(torch.einsum('rd,rdh->h',yp*gate,wo),ywo)
   return o
  _,_,sv=torch.linalg.svd(q,full_matrices=True)
  bases={**fixed,'current_svd_oracle':sv.T}
  for name,a in bases.items():
   qa=q@a;ka=k@a;va=v@a
   if name=='current_svd_oracle':basis_records.append({**meta,'basis':name,'step':step,'definition':'same-step SVD oracle; not deployable persistent compression','sha256':tensor_hash(a)})
   equivalent=qa@ka.T*scale
   exact=float((equivalent-z).abs().max());value_exact=float(((p@va)@a.T-y).abs().max())
   if exact>1e-7*max(1.,float(z.abs().max())) or value_exact>1e-7*max(1.,float(y.abs().max())):raise RuntimeError('Basis algebra gate failed')
   energy=qa.reshape(r,8,32).square().sum((0,2));order=energy.argsort(descending=True,stable=True)
   fixed_order=torch.arange(8,device=device) # stable SVD basis only first blocks; all bases also report query-selected oracle upper bound
   for selection,ix in [('current_query_top_energy',order),('fixed_prefix',fixed_order)]:
    for count in CFG['block_counts']:
     mask=torch.zeros(8,device=device,dtype=torch.float64);mask[ix[:count]]=1;qm=qa*mask.repeat_interleave(32)
     zz=qm@ka.T*scale;tail=center(z-zz)
     emit('blocks',step=step,basis=name,selection=selection,blocks=count,heldout=step>=CFG['basis_train_steps'],energy_capture=float(energy[ix[:count]].sum()/energy.sum().clamp_min(1e-300)),score_abs_error=norm(tail),score_reference_norm=norm(center(z)),score_relative_error=norm(tail)/denom,score_per_qhead=[err(center(zz)[i],center(z)[i]) for i in range(r)],probability_TV_max=float(((zz.softmax(-1)-p).abs().sum(-1)*.5).max()),output=output_stats(zz.softmax(-1)@v),basis_algebra_max_abs=exact,value_algebra_max_abs=value_exact)
   # H32 must preserve each existing block's query energy.
   if name=='block_H32':
    e0=q.reshape(r,8,32).square().sum((0,2));assert torch.allclose(energy,e0,rtol=1e-10,atol=1e-10)
   kh,ko=qdq(ka);vh,vo=qdq(va);zk=qa@kh.T*scale;pk=zk.softmax(-1)
   emit('quantization',step=step,basis=name,K_reconstruction=err(kh,ka),V_reconstruction=err(vh,va),K_centered_scaled_logits=err(center(zk),center(z)),K_probability_TV_max=float(((pk-p).abs().sum(-1)*.5).max()),K_only_output=output_stats(pk@v),V_only_output=output_stats((p@vh)@a.T),joint_output=output_stats((pk@vh)@a.T),K_scale_spectrum=spectrum(ko.scale.double()),V_scale_spectrum=spectrum(vo.scale.double()))
   for kind,x in [('Q',qa),('K',ka),('V',va)]:
    b=x.reshape(-1,8,32);ab=b.abs();nz=torch.where(ab>0,ab,float('inf')).amin(-1);maxv=ab.amax(-1);ratio=maxv/nz
    emit('geometry',step=step,basis=name,operand=kind,coherence_mean=float((x.abs().amax(-1)/x.norm(dim=-1).clamp_min(1e-300)).mean()),block_max_mean=float(maxv.mean()),block_range_mean=float((b.amax(-1)-b.amin(-1)).mean()),block_log2_dynamic_range_mean=float(torch.where(torch.isfinite(ratio)&(ratio>0),ratio.clamp_min(1).log2(),0).mean()))
   qh,ints,info=planes(qa)
   round_z=qh@ka.T*scale;centered_round=center(round_z)
   chosen={b:None for b in CFG['plane_operator_relative_budgets']};round_denom=max(norm(centered_round),1e-30)
   last=None
   for n in range(0,info['nonempty_planes_max']+1):
    rec=truncated_planes(ints,n);zz=rec@ka.T*scale;relative=norm(center(zz-round_z))/round_denom
    for budget in chosen:
     if chosen[budget] is None and relative<=budget:chosen[budget]=n
    emit('planes',step=step,basis=name,planes_kept=n,scaled_centered_logits_rel_to_FP16_transformed=relative,scaled_centered_logits_rel_to_original=norm(center(zz-z))/denom,**({'plane_accounting':info} if n==0 else {}))
    last=rec
   assert torch.equal(last,qh)
   emit('plane_frontier',step=step,basis=name,heldout=step>=CFG['basis_train_steps'],budgets_to_min_planes=chosen,**info)
  # Full-token V support; no token subsampling or renormalization.
  for head in range(r):
   prob=p[head];vals,ix=prob.sort(descending=True,stable=True);cum=vals.cumsum(0)
   emit('support',step=step,qhead_in_group=head,target_mass=1.,support=len(k),support_fraction=1.,N_eff=float(1/prob.square().sum()),delta=0.,bound=0.,actual_error=0.,reference_norm=norm(y[head]))
   for target in CFG['probability_mass_targets']:
    count=min(len(k),int(torch.searchsorted(cum,torch.tensor(target,device=device,dtype=torch.float64)))+1);keep=ix[:count];delta=float(vals[count:].sum());reduced=prob[keep]@v[keep];bound=delta*float(v.norm(dim=-1).max());actual=norm(y[head]-reduced)
    assert actual<=bound+1e-9*max(1.,norm(y[head]))
    row=dict(step=step,qhead_in_group=head,target_mass=target,support=count,support_fraction=count/len(k),N_eff=float(1/prob.square().sum()),delta=delta,bound=bound,actual_error=actual,reference_norm=norm(y[head]),relative_error=actual/max(norm(y[head]),1e-30))
    if wo is not None:row['WO_linear_head_error']=norm(((reduced-y[head])*gate[head])@wo[head]);row['WO_linear_bound']=bound*float(gate[head].abs().max())*norm(wo[head]);row['WO_bound_norm']='Frobenius upper bound'
    emit('support',**row)
  # All counts theoretical, per KV group and step, independent of basis oracle selection.
  L=len(k);tiles_k=8*math.ceil(r/8)*math.ceil(L/8);tiles_v=8*math.ceil(r/8)*math.ceil(32/8)*math.ceil(L/32)
  emit('circuit',step=step,tokens=L,packed_K_bytes=L*d//2,packed_V_bytes=L*d//2,K_scale_bytes=L*8*2,V_scale_bytes=L*8*2,K_logical_width32_dot_products=r*L*8,K_candidate_m8n8k32_tiles_per_Q_plane=tiles_k,K_M_utilization=r/8,K_N_utilization=L/(8*math.ceil(L/8)),V_candidate_m8n8k32_tiles_per_coefficient_plane=tiles_v,V_reduction_dimension='tokens; scales vary with token',K_scale_epilogue_multiplications=r*L*8,K_unsigned_rank1_correction_multiplies=r*L*8,V_prescale_multiplications=r*L*8,V_unsigned_correction_multiplies=r*L*8,FP64_diagnostic_logits_bytes=r*L*8,full_H256_Q_add_sub_estimate=r*d*8,full_H256_K_insert_add_sub_estimate=d*8,block_H32_Q_add_sub_estimate=r*d*5,block_H32_K_insert_add_sub_estimate=d*5,RoPE_placement='all transforms applied post-RoPE; no commutation cost claimed',runtime_SVD_and_K_reprojection='not priced; disqualifies oracle bases as deployed savings',native_instruction_verified=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--device',default='cpu');a=p.parse_args();torch.set_num_threads(4)
 index=json.loads((a.run/'capture_index.json').read_text());manifest=json.loads((a.run/'MANIFEST.json').read_text());tables={};records=[]
 for rec in index['states']:
  assert sha(a.run/rec['file'])==rec['sha256'];s=torch.load(a.run/rec['file'],map_location='cpu',weights_only=True);wmeta=index['weights'][str(rec['layer'])];assert sha(a.run/wmeta['file'])==wmeta['sha256'];w=torch.load(a.run/wmeta['file'],map_location='cpu',weights_only=True)
  assert s['q'].shape==(rec['steps'],len(rec['kv_heads']),6,256)
  for j,h in enumerate(rec['kv_heads']):
   meta={k:rec[k] for k in ['domain','partition','context','layer']};meta['kv_head']=h
   wo=w[:,h*6*256:(h+1)*6*256].T.reshape(6,256,-1).to(a.device,dtype=torch.float64)
   analyze_group(s['q'][:,j].to(a.device,dtype=torch.float64),s['base_k'][j].to(a.device,dtype=torch.float64),s['base_v'][j].to(a.device,dtype=torch.float64),s['new_k'][:,j].to(a.device,dtype=torch.float64),s['new_v'][:,j].to(a.device,dtype=torch.float64),wo,s['scaling'],meta,tables,records,gates=s['gate'][:,j].to(a.device,dtype=torch.float64))
   print('ANALYZED',meta,flush=True)
 for name,rows in tables.items():csv_write(a.run/(name+'.csv'),rows)
 (a.run/'basis_definitions.json').write_text(json.dumps({'aliases':CFG['basis_aliases'],'records':records},indent=2))
 result={'execution_status':'ANALYSIS_COMPLETE','scientific_gate':'SMOKE_ONLY' if manifest['smoke'] else 'OBSERVATIONAL_NO_E15_ACCEPTANCE','source_sha':manifest['git_sha'],'analysis_git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'analysis_script_sha256':sha(__file__),'counts':{k:len(v) for k,v in tables.items()},'basis_count':len(records),'all_token_probabilities':True,'WO_scope':CFG['WO_scope']+'; captured native output gate held fixed','latency_claim':False}
 (a.run/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
