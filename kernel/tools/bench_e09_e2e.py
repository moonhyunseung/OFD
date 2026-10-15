"""Actual full-model online-cache decode; original and matched FP16 controls."""
import argparse,copy,csv,gc,hashlib,itertools,json,math,random,time,traceback
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right
from transformers import AutoTokenizer,AutoModelForCausalLM,AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from ofd_kernel.runtime.online_cache import install,move_cache,OnlineKV
from ofd_kernel.contracts import Variant
from ofd_kernel.reference.quant import quantize_int4_packed_groupwise
import bench_e09_layers as source_data
ROOT=Path(__file__).resolve().parents[2]
C=json.loads((ROOT/'kernel/experiments/E09_real_model/e2e_protocol.json').read_text())

class Sweep:
    def __init__(self,out,smoke):
        self.out=out;self.smoke=smoke;self.active=None;self.rows=[];self.steps=4 if smoke else C['steps'];self.warm=1 if smoke else C['warmup_steps'];self.refs={}
        torch.manual_seed(C['seed'])
        def attention(module,q,k,v,mask,dropout=0.,scaling=None,**kw):
            if self.active is not None and module.layer_idx in self.active and isinstance(self.active[module.layer_idx],OnlineKV):return self.active[module.layer_idx].attention(q),None
            if q.shape[2]>1:
                bias=None if q.shape[2]==k.shape[2] else causal_lower_right(q.shape[2],k.shape[2])
                y=F.scaled_dot_product_attention(q,k,v,attn_mask=bias,is_causal=bias is None,enable_gqa=True,scale=scaling)
                return y.transpose(1,2),None
            return sdpa_attention_forward(module,q,k,v,None,dropout=0.,scaling=scaling,**kw)
        AttentionInterface.register('ofd_online',attention)
        self.tok=AutoTokenizer.from_pretrained(C['model_id'],revision=C['model_revision'],trust_remote_code=False)
        self.items={};original=source_data.C
        for spec in (original['datasets'][:1] if smoke else original['datasets']):
            source_data.C={**original,'datasets':[spec],'contexts':[131072 if spec['domain']=='prose' else 32768],'decode_steps':self.steps+1}
            for x in source_data.data(self.tok,smoke):self.items[(x['domain'],x['phase'])]=x
        source_data.C=original
        (out/'data_manifest.json').write_text(json.dumps([{k:v for k,v in x.items() if k!='tokens'} for x in self.items.values()],indent=2)+'\n')
        self.model,loading=AutoModelForCausalLM.from_pretrained(C['model_id'],revision=C['model_revision'],dtype=torch.float16,device_map='cuda',attn_implementation='ofd_online',trust_remote_code=False,output_loading_info=True)
        if loading.get('missing_keys'):raise RuntimeError(str(loading['missing_keys'][:8]))
        self.model.eval();self.model_bytes=sum(p.numel()*p.element_size() for p in self.model.parameters())
        self.f=(out/'e2e.csv').open('w',newline='');self.writer=None
    def template(self,domain,partition,t,b):
        self.active=None;self.base=None;gc.collect();torch.cuda.empty_cache()
        limit=Path('/sys/fs/cgroup/memory.max');stat=Path('/sys/fs/cgroup/memory.stat')
        if limit.exists() and limit.read_text().strip()!='max':
            stats=dict(line.split() for line in stat.read_text().splitlines());budget=int(limit.read_text())-int(stats.get('anon',0))-int(stats.get('kernel',0))
            required=int(2.25*b*(t*65536+48*48*128*128*4))
            if required>0.8*budget:raise MemoryError(f'HOST_TEMPLATE_LIMIT required={required} budget={budget}')
        self.active=None;item=self.items[(domain,partition)];self.case=(domain,partition,t,b)
        ids=torch.tensor([item['tokens'][:t]],device='cuda').expand(b,-1).contiguous();cache=None
        with torch.inference_mode():
            for start in range(0,t,C['prefill_chunk']):
                result=self.model(input_ids=ids[:,start:min(start+C['prefill_chunk'],t)],past_key_values=cache,use_cache=True,logits_to_keep=1)
                cache=result.past_key_values;del result
        move_cache(cache,'cpu');del ids;gc.collect();torch.cuda.empty_cache()
        self.base=cache
        for lyr in cache.layers:
            if any(isinstance(v,torch.Tensor) and v.is_cuda for v in vars(lyr).values()):raise RuntimeError('GPU template retained')
        self.inputs=[torch.full((b,1),item['tokens'][t+s],device='cuda',dtype=torch.long) for s in range(self.steps)]
        self.positions=[torch.full((b,1),t+s,device='cuda',dtype=torch.long) for s in range(self.steps)]
        self.targets=torch.tensor([item['tokens'][t+s+1] for s in range(self.warm,self.steps)],device='cuda')[:,None].expand(-1,b)
        self.mask={'full_attention':None,'linear_attention':None}
    def run(self,policy,label,round=0,layer=-1,variant='-',phase='control',unmodified=False):
        domain,partition,t,b=self.case;cache=move_cache(copy.deepcopy(self.base),'cuda');self.active=None if unmodified else install(cache,policy,self.steps)
        kvbytes=sum(x.kv_bytes for x in self.active.values()) if self.active else sum(cache.layers[i].keys.numel()*cache.layers[i].keys.element_size()*2 for i in C['layers'])
        events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(self.steps-self.warm)]
        for ev_start,ev_end in events:ev_start.record();ev_end.record()
        torch.cuda.synchronize()
        logits=[];wall_start=0
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();allocated=torch.cuda.memory_allocated()
        with torch.inference_mode():
            for s in range(self.steps):
                if s==self.warm:torch.cuda.synchronize();wall_start=time.perf_counter()
                if s>=self.warm:events[s-self.warm][0].record()
                result=self.model(input_ids=self.inputs[s],position_ids=self.positions[s],attention_mask=self.mask,past_key_values=cache,use_cache=True,logits_to_keep=1)
                if s>=self.warm:events[s-self.warm][1].record();logits.append(result.logits[:,0,:])
                cache=result.past_key_values;del result
        torch.cuda.synchronize();wall=(time.perf_counter()-wall_start)*1000/(self.steps-self.warm)
        ms=sum(a.elapsed_time(z) for a,z in events)/(self.steps-self.warm);peak=torch.cuda.max_memory_allocated()
        y=torch.stack(logits).float()
        if not torch.isfinite(y).all().item():raise RuntimeError('Non-finite model logits')
        nll=F.cross_entropy(y.reshape(-1,y.shape[-1]),self.targets.reshape(-1)).item()
        refkey=(domain,partition,t,b)
        if label=='NATIVE' and round==0:self.refs[refkey]=y.cpu()
        ref=self.refs.get(refkey);err=float('nan');agreement=float('nan')
        if ref is not None:
            rg=ref.to(y.device);err=((y-rg).norm()/(rg.norm()+1e-12)).item();agreement=(y.argmax(-1)==rg.argmax(-1)).float().mean().item();del rg
        row=dict(phase=phase,domain=domain,partition=partition,context=t,batch=b,round=round,policy_label=label,changed_layer=layer,variant=variant,policy_json=json.dumps(policy,sort_keys=True),decode_ms=wall,gpu_event_ms=ms,aggregate_tok_s=b*1000/wall,gpu_event_tok_s=b*1000/ms,wall_ms=wall,kv_bytes=kvbytes,allocated_bytes=allocated,peak_allocated_bytes=peak,model_parameter_bytes=self.model_bytes,logit_rel_l2=err,argmax_agreement=agreement,nll=nll,reference_available=ref is not None,status='OK',seed=C['seed'])
        self.write(row)
        self.active=None;del cache,y,logits;gc.collect();torch.cuda.empty_cache()
        return row
    def write(self,row):
        if self.writer is None:self.writer=csv.DictWriter(self.f,fieldnames=list(row));self.writer.writeheader()
        self.writer.writerow(row);self.f.flush();self.rows.append(row);print('E2E', {k:row[k] for k in ['phase','domain','context','batch','policy_label','changed_layer','variant','decode_ms','status']},flush=True)
    def matrix(self):
        layers=C['layers'];uniform=lambda v:{str(i):v for i in layers}
        if self.smoke:
            self.template('prose','calibration',128,1)
            self.run(uniform('NATIVE'),'NATIVE')
            raw=self.run(uniform('NATIVE'),'NATIVE_UNMODIFIED',unmodified=True)
            if raw['logit_rel_l2']>0.002:raise RuntimeError('Native cache replacement correctness failed')
            fp=self.run(uniform('FP16'),'FP16')
            if fp['logit_rel_l2']>0.002:raise RuntimeError('Paged FP16 model correctness failed')
            for v in C['variants']:self.run(uniform(v),v)
            mixed=uniform('P');mixed[str(layers[0])]='K';mixed[str(layers[1])]='V';self.run(mixed,'MIX_K_V')
            self.f.close();return
        calibration=[]
        for domain in C['calibration']['domains']:
            self.template(domain,'calibration',8192,1);self.run(uniform('NATIVE'),'NATIVE')
            self.run(uniform('FP16'),'FP16');self.run(uniform('P'),'P');self.run(uniform('KV'),'KV')
            for rnd in range(C['calibration']['rounds']):
                order=list(itertools.product(layers,C['variants']));random.Random(C['seed']+rnd).shuffle(order)
                for layer,v in order:
                    policy=uniform('P');policy[str(layer)]=v
                    calibration.append(self.run(policy,'ONE_LAYER',rnd,layer,v,'calibration'))
        selected={}
        for layer in layers:
            score={v:math.exp(sum(math.log(x['decode_ms']) for x in calibration if x['changed_layer']==layer and x['variant']==v)/sum(x['changed_layer']==layer and x['variant']==v for x in calibration)) for v in C['variants']}
            selected[str(layer)]=min(score,key=score.get)
        (self.out/'speed_policy.json').write_text(json.dumps(dict(policy=selected,frozen_before_evaluation=True,selection='complete_model_decode_latency',calibration_count=len(calibration)),indent=2)+'\n')
        print('FROZEN_SPEED_POLICY',selected,flush=True)
        shapes=[('evaluation',d,t,1) for d in C['evaluation']['domains'] for t in C['evaluation']['contexts']]
        shapes += [('crossover','prose',t,b) for t in C['crossover']['contexts'] for b in C['crossover']['batches']]
        for phase,domain,t,b in shapes:
            try:self.template(domain,'evaluation',t,b)
            except (torch.OutOfMemoryError,MemoryError) as error:
                print('SETUP_OOM',phase,domain,t,b,flush=True);(self.out/f'OOM_{phase}_{domain}_{t}_{b}.json').write_text(json.dumps(dict(phase=phase,domain=domain,context=t,batch=b,status='PREFILL_SETUP_OOM' if isinstance(error,torch.OutOfMemoryError) else 'HOST_TEMPLATE_LIMIT',reason=str(error)))+'\n');gc.collect();torch.cuda.empty_cache();continue
            choices=[('NATIVE',uniform('NATIVE')),('FP16',uniform('FP16')),('P',uniform('P')),('KV',uniform('KV')),('MIX',selected)]
            for rnd in range(3):
                order=choices.copy()
                if rnd:random.Random(C['seed']+rnd).shuffle(order)
                for label,policy in order:
                    try:self.run(policy,label,rnd,phase=phase)
                    except torch.OutOfMemoryError:
                        print('CASE_OOM',phase,domain,t,b,label,rnd,flush=True);(self.out/f'OOM_{phase}_{domain}_{t}_{b}_{label}_{rnd}.json').write_text(json.dumps(dict(status='DECODE_SETUP_OR_EXECUTION_OOM',policy=policy))+'\n');gc.collect();torch.cuda.empty_cache()
            self.refs.clear()
        self.f.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    Sweep(a.out,a.smoke).matrix();(a.out/'COMPLETED.json').write_text(json.dumps(dict(completed=True,smoke=a.smoke,scope='full_model_teacher_forced_online_decode',prefill_timed=False))+'\n')
if __name__=='__main__':main()
