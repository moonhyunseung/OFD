"""Real Qwen teacher-forced decode layer replay. Never claims serving E2E."""
import argparse,csv,gc,hashlib,json,math,sys,zipfile
from huggingface_hub import hf_hub_download
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForCausalLM,AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from ofd_kernel.contracts import Variant
from ofd_kernel.reference.fwht import fwht_normalized
from ofd_kernel.reference.quant import quantize_int4_packed_groupwise
from ofd_kernel.runtime.launcher import TritonDecodeConfig,allocate_workspace
from ofd_kernel.runtime.fp16_launcher import fp16_paged_decode_core
from ofd_kernel.runtime.gpu_ofd import gpu_ofd_decode
from bench_gpu import cuda_ms,rel
ROOT=Path(__file__).resolve().parents[2]
C=json.loads((ROOT/'kernel/experiments/E09_real_model/decode_protocol.json').read_text())

def rows_text(ds):
    for i,x in enumerate(ds):
        if 'context' in x: text=x['context']+'\n'+x.get('input','')
        elif 'question' in x:text=x['question']+'\n'+x.get('answer','')
        elif 'code' in x:text='\n'.join([x.get('text',''),x.get('test_setup_code',''),x['code'],*x.get('test_list',[]),*x.get('challenge_test_list',[])])
        else:text=x.get('text','')
        if text.strip():yield i,text

def data(tokenizer,smoke):
    items=[]
    for spec in C['datasets'][:1] if smoke else C['datasets']:
        for phase in ['calibration','evaluation']:
            split=spec[phase+'_split']
            if spec['id']=='THUDM/LongBench':
                archive=hf_hub_download(spec['id'],'data.zip',repo_type='dataset',revision=spec['revision'])
                with zipfile.ZipFile(archive) as z:
                    name=next(n for n in z.namelist() if n.endswith('/qasper.jsonl') or n=='qasper.jsonl')
                    ds=[json.loads(line) for line in z.read(name).decode().splitlines() if line.strip()]
            else:ds=load_dataset(spec['id'],spec['subset'],revision=spec['revision'],split=split)
            tokens=[];ids=[];needed=(256 if smoke else max(C['contexts']))+C['decode_steps']
            for idx,txt in rows_text(ds):
                if 'partition' in spec and int(hashlib.sha256(txt.encode()).hexdigest(),16)%2 != (0 if phase=='calibration' else 1):continue
                tokens.extend(tokenizer.encode(txt,add_special_tokens=False)+[tokenizer.eos_token_id]);ids.append(idx)
                if len(tokens)>=needed:break
            if len(tokens)<needed:raise RuntimeError(f'Insufficient unique data: {spec["id"]} {phase} {len(tokens)}')
            items.append(dict(domain=spec['domain'],phase=phase,tokens=tokens[:needed],source=spec,source_row_ids=ids,token_sha256=hashlib.sha256(torch.tensor(tokens[:needed],dtype=torch.int64).numpy().tobytes()).hexdigest()))
    return items

def page(x,perm):
    b,h,t,d=x.shape;n=(t+15)//16
    padded=torch.zeros((b,h,n*16,d),device=x.device,dtype=x.dtype);padded[:,:,:t].copy_(x)
    logical=padded.reshape(b,h,n,16,d).permute(0,2,1,3,4).reshape(b*n,h,16,d).contiguous()
    physical=torch.empty_like(logical);physical[perm]=logical
    return physical

def packed(x):
    p,h,t,d=x.shape;o=quantize_int4_packed_groupwise(x,32)
    return o.packed.reshape(p,h,t,d//32,16),o.scale.reshape(p,h,t,d//32)

def compare(q,k,v,output):
    q=q[:,:,0,:].contiguous();b,hq,d=q.shape;t=k.shape[2]
    perm=torch.randperm(b*((t+15)//16),device=q.device)
    bt=perm.reshape(b,-1).to(torch.int32);lens=torch.full((b,),t,device=q.device,dtype=torch.int32)
    kpage=page(k,perm);vpage=page(v,perm);cfg=TritonDecodeConfig(**C['kernel_config']);ws=allocate_workspace(q,lens,cfg,max_seq_len=t)
    fp=lambda:fp16_paged_decode_core(q,kpage,vpage,bt,lens,config=cfg,workspace=ws,max_seq_len=t)
    reference=fp().clone();err=rel(reference,output.transpose(1,2)[:,:,0,:])
    if not math.isfinite(err) or err>0.002:raise RuntimeError(f'FP16 paged/native mismatch {err}')
    fm=cuda_ms(fp,3,10);rows=[]
    for variant in Variant:
        kk=fwht_normalized(kpage) if variant.rotate_k else kpage
        vv=fwht_normalized(vpage) if variant.rotate_v else vpage
        kp,ks=packed(kk);vp,vs=packed(vv);qw=torch.empty_like(q);ow=torch.empty_like(q)
        fn=lambda:gpu_ofd_decode(q,kp,ks,vp,vs,bt,lens,variant=variant,config=cfg,workspace=ws,max_seq_len=t,q_work=qw,out_work=ow)
        target=fn();quality=rel(target,reference);ms=cuda_ms(fn,3,10)
        if not math.isfinite(quality):raise RuntimeError('Nonfinite INT4 output')
        rows.append(dict(variant=variant.value,fp16_ms=fm,int4_ms=ms,throughput_ratio=fm/ms,relative_l2_error=quality,fp16_native_error=err,fp16_kv_bytes=sum(x.numel()*x.element_size() for x in [kpage,vpage]),int4_kv_bytes=sum(x.numel()*x.element_size() for x in [kp,ks,vp,vs]),kv_context=t))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.manual_seed(C['seed']);a.out.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(C['model_id'],revision=C['model_revision'],trust_remote_code=False)
    items=data(tok,a.smoke)
    (a.out/'data_manifest.json').write_text(json.dumps([{k:v for k,v in x.items() if k!='tokens'} for x in items],indent=2)+'\n')
    fields=['domain','phase','context','decode_step','layer','variant','fp16_ms','int4_ms','throughput_ratio','relative_l2_error','fp16_native_error','fp16_kv_bytes','int4_kv_bytes','kv_context']
    allrows=[];active={}
    with (a.out/'layer_decode.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        def backend(module,q,k,v,mask,dropout=0.0,scaling=None,**kw):
            output,weights=sdpa_attention_forward(module,q,k,v,mask,dropout=dropout,scaling=scaling,**kw)
            if q.shape[2]==1 and active.get('measure'):
                for row in compare(q,k,v,output):
                    row.update(**active['case'],layer=module.layer_idx);writer.writerow(row);allrows.append(row)
                f.flush();print('LAYER',active['case'],module.layer_idx,flush=True)
            return output,weights
        AttentionInterface.register('ofd_layer_probe',backend)
        model,loading=AutoModelForCausalLM.from_pretrained(C['model_id'],revision=C['model_revision'],dtype=torch.float16,device_map='cuda',attn_implementation='ofd_layer_probe',trust_remote_code=False,output_loading_info=True)
        if loading.get('missing_keys'):raise RuntimeError(f'Missing model weights: {loading["missing_keys"][:8]}')
        model.eval()
        for phase in ['calibration','evaluation']:
            if phase=='evaluation':
                policy={}
                for layer in C['full_attention_layers']:
                    means={v.value:sum(x['relative_l2_error'] for x in allrows if x['layer']==layer and x['phase']=='calibration' and x['variant']==v.value)/max(1,sum(x['layer']==layer and x['phase']=='calibration' and x['variant']==v.value for x in allrows)) for v in Variant}
                    policy[str(layer)]=min(means,key=means.get)
                (a.out/'policy.json').write_text(json.dumps(dict(policy=policy,frozen_before_evaluation=True,scope='replay_policy_only'),indent=2)+'\n')
            for item in [x for x in items if x['phase']==phase]:
                for t in ([256] if a.smoke else C['contexts']):
                    ids=torch.tensor([item['tokens'][:t]],device='cuda');active['measure']=False
                    with torch.inference_mode():
                        pre=model(input_ids=ids,use_cache=True,logits_to_keep=1);cache=pre.past_key_values;del pre
                        for step in range(1 if a.smoke else C['decode_steps']):
                            active.update(measure=step in ([0] if a.smoke else C['measured_decode_steps']),case=dict(domain=item['domain'],phase=phase,context=t,decode_step=step))
                            result=model(input_ids=torch.tensor([[item['tokens'][t+step]]],device='cuda'),past_key_values=cache,use_cache=True,logits_to_keep=1);cache=result.past_key_values;del result
                    del cache,ids;gc.collect();torch.cuda.empty_cache()
    expected=(1 if a.smoke else 4)*2*(1 if a.smoke else 3)*(1 if a.smoke else 2)*16*4
    if len(allrows)!=expected:raise RuntimeError(f'Expected {expected} rows, got {len(allrows)}')
    (a.out/'summary.json').write_text(json.dumps(dict(rows=len(allrows),scope='real_model_decode_attention_replay_only',e09_overall_complete=False,online_append_measured=False,model_total_latency_measured=False,smoke=a.smoke),indent=2)+'\n')
if __name__=='__main__':main()
