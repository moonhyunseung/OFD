from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch

from ofd_kernel.bench import BenchmarkCase, full_matrix, smoke_matrix, long_context_matrix
from ofd_kernel.contracts import Variant
from ofd_kernel.reference.paged import pack_dense_to_paged_int4
from ofd_kernel.reference.attention import paged_quantized_variant_attention
from ofd_kernel.runtime.launcher import (
    TritonDecodeConfig,
    allocate_workspace,
    prepare_query,
    triton_paged_decode,
    triton_paged_decode_core,
)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / (a.float().norm() + 1e-12)).item()


def cuda_ms(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def to_cuda_cache(cache, device: torch.device):
    return (
        cache.k_packed.to(device),
        cache.k_scale.to(device),
        cache.v_packed.to(device),
        cache.v_scale.to(device),
        cache.block_table.to(device=device, dtype=torch.int32),
        cache.seq_lens.to(device=device, dtype=torch.int32),
    )


def correctness(device: torch.device, config: TritonDecodeConfig, threshold: float = 2e-3, seed: int = 1337) -> None:
    # Force multiple splits so short requests exercise explicit empty-split
    # states on the very first GPU correctness pass.
    config = TritonDecodeConfig(
        split_size=min(config.split_size, 64),
        tokens_per_block=min(config.tokens_per_block, 16),
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    b, hq, hkv, t, d = 3, 8, 2, 257, 64
    seq_lens = torch.tensor([17, 93, 257], dtype=torch.long)
    q = torch.randn(b, hq, d, dtype=torch.float16)
    k = torch.randn(b, hkv, t, d, dtype=torch.float16)
    v = torch.randn(b, hkv, t, d, dtype=torch.float16)

    print("GPU CORRECTNESS: randomized physical pages + ragged lengths")
    print("correctness config:", config)
    worst = 0.0
    for variant in Variant:
        ref, cache = paged_quantized_variant_attention(
            q,
            k,
            v,
            variant,
            seq_lens=seq_lens,
            group_size=32,
            page_size=16,
            split_size=config.split_size,
            seed=20261012,
        )
        kp, ks, vp, vs, bt, lens = to_cuda_cache(cache, device)
        qg = q.to(device)
        ws = allocate_workspace(qg, lens, config, max_seq_len=t)
        got = triton_paged_decode(
            qg,
            kp,
            ks,
            vp,
            vs,
            bt,
            lens,
            variant=variant,
            config=config,
            workspace=ws,
            max_seq_len=t,
            debug_validate=True,
        ).cpu()
        e = rel(ref, got)
        worst = max(worst, e)
        print(f"  {variant.value:>2}: rel={e:.6e}")

    print(f"worst relative error: {worst:.6e}")
    print(f"frozen threshold: {threshold:.6e}")
    if (not math.isfinite(worst)) or worst > threshold:
        raise RuntimeError(f"E02 correctness gate failed: worst={worst:.6e} > {threshold:.6e}")
    print("E02_CORRECTNESS_GATE_PASS")


def make_random_paged_case(
    case: BenchmarkCase,
    *,
    num_kv_heads: int,
    device: torch.device,
    seed: int,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    b, t, d = case.batch, case.context, case.head_dim
    hkv = num_kv_heads
    hq = hkv * case.q_per_kv
    page = case.page_size
    group = case.group_size
    groups = d // group
    bpg = group // 2
    pages_per_req = math.ceil(t / page)
    total_pages = b * pages_per_req

    q = torch.randn((b, hq, d), device=device, dtype=torch.float16)
    kp = torch.randint(
        0, 256, (total_pages, hkv, page, groups, bpg),
        device=device, dtype=torch.uint8
    )
    vp = torch.randint_like(kp, 0, 256)
    ks = torch.rand((total_pages, hkv, page, groups), device=device, dtype=torch.float16) * 0.08 + 0.005
    vs = torch.rand_like(ks) * 0.08 + 0.005

    # Each request owns disjoint pages; logical->physical order is shuffled.
    perm = torch.randperm(total_pages, device=device, dtype=torch.int64)
    bt = perm.reshape(b, pages_per_req).to(torch.int32)
    lens = torch.full((b,), t, device=device, dtype=torch.int32)
    return q, kp, ks, vp, vs, bt, lens


def benchmark_cases(
    device: torch.device,
    cases: list[BenchmarkCase],
    out_csv: Path,
    config: TritonDecodeConfig,
    warmup: int,
    repeat: int,
    num_kv_heads: int,
    seed: int,
    core_only: bool = False,
) -> None:
    fields = [
        "gpu", "torch", "context", "batch", "q_per_kv", "hkv", "hq",
        "head_dim", "page_size", "group_size", "variant", "split_size",
        "tokens_per_block", "num_warps", "num_stages", "latency_ms",
        "aggregate_tok_s", "seed", "timing_scope",
    ]
    rows = []

    for idx, case in enumerate(cases, 1):
        print(f"[{idx}/{len(cases)}] T={case.context} B={case.batch} Q/KV={case.q_per_kv}")
        case_seed = seed + idx
        q, kp, ks, vp, vs, bt, lens = make_random_paged_case(
            case, num_kv_heads=num_kv_heads, device=device, seed=case_seed
        )
        ws = allocate_workspace(q, lens, config, max_seq_len=case.context)

        for variant in Variant:
            if core_only:
                q_work = prepare_query(q, variant).contiguous()
                fn = lambda q_work=q_work: triton_paged_decode_core(
                    q_work, kp, ks, vp, vs, bt, lens,
                    config=config, workspace=ws, max_seq_len=case.context,
                )
                timing_scope = "core_attention_only"
            else:
                fn = lambda variant=variant: triton_paged_decode(
                    q, kp, ks, vp, vs, bt, lens, variant=variant,
                    config=config, workspace=ws, max_seq_len=case.context,
                )
                timing_scope = "end_to_end_reference_fwht"
            ms = cuda_ms(fn, warmup, repeat)
            rows.append({
                "gpu": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "context": case.context,
                "batch": case.batch,
                "q_per_kv": case.q_per_kv,
                "hkv": num_kv_heads,
                "hq": num_kv_heads * case.q_per_kv,
                "head_dim": case.head_dim,
                "page_size": case.page_size,
                "group_size": case.group_size,
                "variant": variant.value,
                "split_size": config.split_size,
                "tokens_per_block": config.tokens_per_block,
                "num_warps": config.num_warps,
                "num_stages": config.num_stages,
                "latency_ms": ms,
                "aggregate_tok_s": case.batch * 1000.0 / ms,
                "seed": case_seed,
                "timing_scope": timing_scope,
            })
            print(f"   {variant.value:>2}: {ms:.4f} ms | {case.batch*1000.0/ms:.1f} tok/s")

        del q, kp, ks, vp, vs, bt, lens, ws
        torch.cuda.empty_cache()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("CSV:", out_csv)



def compile_smoke(device: torch.device, config: TritonDecodeConfig, seed: int = 1337) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    b, hq, hkv, t, d = 1, 2, 1, 64, 64
    q = torch.randn(b, hq, d, dtype=torch.float16)
    k = torch.randn(b, hkv, t, d, dtype=torch.float16)
    v = torch.randn(b, hkv, t, d, dtype=torch.float16)
    lens = torch.tensor([t], dtype=torch.long)
    _, cache = paged_quantized_variant_attention(
        q, k, v, Variant.P, seq_lens=lens, group_size=32, page_size=16,
        split_size=64, seed=1337,
    )
    kp, ks, vp, vs, bt, lens_g = to_cuda_cache(cache, device)
    qg = q.to(device)
    ws = allocate_workspace(qg, lens_g, config, max_seq_len=t)
    out = triton_paged_decode(
        qg, kp, ks, vp, vs, bt, lens_g, variant=Variant.P, config=config,
        workspace=ws, max_seq_len=t, debug_validate=True,
    )
    torch.cuda.synchronize()
    finite = bool(torch.isfinite(out).all().item())
    print("COMPILE_SMOKE", tuple(out.shape), finite)
    if not finite:
        raise RuntimeError("E01 compile smoke produced non-finite output")
    print("E01_COMPILE_GATE_PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("compile", "correctness", "core-variants", "variants", "long-core", "long", "full"), default="correctness")
    ap.add_argument("--out", default="kernel/results/bench.csv")
    ap.add_argument("--split-size", type=int, default=1024)
    ap.add_argument("--tokens-per-block", type=int, default=16)
    ap.add_argument("--num-warps", type=int, default=4)
    ap.add_argument("--num-stages", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument("--num-kv-heads", type=int, default=8)
    ap.add_argument("--correctness-threshold", type=float, default=2e-3)
    ap.add_argument("--kv-plain-max-ratio", type=float, default=1.05)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is unavailable. No GPU work was started.")
        raise SystemExit(3)

    device = torch.device("cuda:0")
    cfg = TritonDecodeConfig(
        split_size=args.split_size,
        tokens_per_block=args.tokens_per_block,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )
    from ofd_kernel.triton.configs import BLACKWELL_CANDIDATES
    print(json.dumps({
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "mode": args.mode,
        "config": cfg.__dict__,
        "blackwell_candidates": [c.__dict__ for c in BLACKWELL_CANDIDATES],
    }, indent=2))

    if args.mode == "compile":
        compile_smoke(device, cfg, args.seed)
    elif args.mode == "correctness":
        correctness(device, cfg, args.correctness_threshold, args.seed)
    else:
        cases = smoke_matrix() if args.mode in ("core-variants", "variants") else long_context_matrix() if args.mode in ("long-core", "long") else full_matrix()
        out_path = Path(args.out)
        core_only = args.mode in ("core-variants", "long-core")
        benchmark_cases(
            device, cases, out_path, cfg, args.warmup, args.repeat,
            args.num_kv_heads, args.seed, core_only=core_only,
        )
        if args.mode in ("core-variants", "variants"):
            import csv as _csv
            rows = list(_csv.DictReader(out_path.open()))
            groups = {}
            for r in rows:
                key = (r["context"], r["batch"], r["q_per_kv"], r["hkv"], r["head_dim"], r["page_size"], r["group_size"])
                groups.setdefault(key, {})[r["variant"]] = float(r["latency_ms"])
            ratios = []
            for key, vals in groups.items():
                if "P" not in vals or "KV" not in vals:
                    raise RuntimeError(f"missing P/KV rows for {key}")
                ratio = vals["KV"] / vals["P"]
                ratios.append(ratio)
                print(f"E04 ratio {key}: KV/P={ratio:.4f}")
            worst = max(ratios) if ratios else float("inf")
            print(f"E04 worst KV/P={worst:.4f}, frozen max={args.kv_plain_max_ratio:.4f}")
            if (not math.isfinite(worst)) or worst > args.kv_plain_max_ratio:
                raise RuntimeError(f"E04 gate failed: worst KV/P={worst:.4f}")
            print("E04_CORE_ATTENTION_GATE_PASS" if args.mode == "core-variants" else "REFERENCE_E2E_RATIO_PASS")

if __name__ == "__main__":
    main()
