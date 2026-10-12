"""Historical v1 serial/paged INT4 versus v2 split-KV core.

Both paths read identical packed bytes and scales. Layout conversion and
workspace allocation happen before timing. The v1 kernel is copied verbatim
from the archived 2026-10-11 prototype; its bf16 output dictates a shared
bf16 query/output dtype for this comparison.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
from pathlib import Path
import torch

from ofd_kernel.bench import BenchmarkCase
from ofd_kernel.runtime.launcher import TritonDecodeConfig, allocate_workspace, triton_paged_decode_core
from bench_gpu import cuda_ms, make_random_paged_case

HISTORICAL = Path(__file__).resolve().parents[1] / 'src/ofd_kernel/historical_v1/benchmark_paged_ofd_triton.py'
spec = importlib.util.spec_from_file_location('ofd_historical_v1', HISTORICAL)
v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)


def run(out: Path, smoke: bool, warmup: int, repeat: int) -> None:
    device = torch.device('cuda:0')
    cfg = TritonDecodeConfig()
    contexts = [8192] if smoke else [8192, 32768, 65536, 131072]
    batches = [1, 4] if smoke else [1, 4, 8, 16, 32]
    fields = ['context','batch','q_per_kv','hkv','head_dim','page_size','group_size','dtype',
              'v1_ms','v2_ms','v1_tok_s','v2_tok_s','v2_v1_throughput_ratio',
              'v1_kv_bytes','v2_kv_bytes','v1_v2_rel_error','v1_source_sha256','status','seed','timing_scope']
    import hashlib
    source_hash = hashlib.sha256(HISTORICAL.read_bytes()).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for context in contexts:
            for batch in batches:
                case = BenchmarkCase(context,batch,1)
                seed = 1337 + context + batch
                row = dict(context=context,batch=batch,q_per_kv=1,hkv=8,head_dim=128,
                           page_size=16,group_size=32,dtype='bf16',seed=seed,
                           v1_source_sha256=source_hash,timing_scope='core_attention_only',status='OK')
                try:
                    q, kp, ks, vp, vs, bt, lens = make_random_paged_case(case,num_kv_heads=8,device=device,seed=seed)
                    q = q.to(torch.bfloat16)
                    # Same data, physical page mapping and quantizer contract;
                    # historical layout [P,T,H,bytes] is constructed untimed.
                    pk1 = kp.permute(0,2,1,3,4).contiguous().view(-1,16,8,64)
                    pv1 = vp.permute(0,2,1,3,4).contiguous().view(-1,16,8,64)
                    sk1 = ks.permute(0,2,1,3).contiguous()
                    sv1 = vs.permute(0,2,1,3).contiguous()
                    out1 = torch.empty_like(q)
                    ws = allocate_workspace(q,lens,cfg,max_seq_len=context)
                    def old():
                        v1.paged_ofd_int4_decode_kernel[(batch,8)](
                            q,pk1,sk1,pv1,sv1,bt,lens,out1,bt.stride(0),
                            H=8,D=128,P=16,GROUP=32,N_GROUPS=4,BLOCK_D=128,
                            SM_SCALE=1/math.sqrt(128),num_warps=4,num_stages=1)
                    def new():
                        triton_paged_decode_core(q,kp,ks,vp,vs,bt,lens,
                                                  config=cfg,workspace=ws,max_seq_len=context)
                    # Numerical check before timing; threshold includes bf16 output rounding.
                    old(); new(); torch.cuda.synchronize()
                    err = ((out1.float()-ws.out.float()).norm()/(ws.out.float().norm()+1e-12)).item()
                    if not math.isfinite(err) or err > 0.01:
                        raise RuntimeError(f'v1/v2 shared-cache correctness error {err:.6g}')
                    row['v1_v2_rel_error'] = err
                    row['v1_ms'] = cuda_ms(old,warmup,repeat)
                    row['v2_ms'] = cuda_ms(new,warmup,repeat)
                    row['v1_tok_s'] = batch*1000/row['v1_ms']
                    row['v2_tok_s'] = batch*1000/row['v2_ms']
                    row['v2_v1_throughput_ratio'] = row['v2_tok_s']/row['v1_tok_s']
                    row['v1_kv_bytes'] = sum(x.numel()*x.element_size() for x in (pk1,sk1,pv1,sv1))
                    row['v2_kv_bytes'] = sum(x.numel()*x.element_size() for x in (kp,ks,vp,vs))
                    assert row['v1_kv_bytes']==row['v2_kv_bytes']
                    print(context,batch,round(row['v1_ms'],4),round(row['v2_ms'],4),round(row['v2_v1_throughput_ratio'],3),err,flush=True)
                except torch.OutOfMemoryError:
                    row['status']='OOM'
                    print(context,batch,'OOM',flush=True)
                writer.writerow({k:row.get(k,'') for k in fields})
                f.flush()
                q=kp=ks=vp=vs=bt=lens=pk1=pv1=sk1=sv1=out1=ws=None
                torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--repeat',type=int,default=10)
    a=p.parse_args()
    run(a.out,a.smoke,a.warmup,a.repeat)
