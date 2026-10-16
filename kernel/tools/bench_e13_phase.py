"""E13: phase attribution for the frozen subgrouped path.

No hot-kernel optimization is introduced here. The script launches the existing
subgrouped Stage-1 and existing split merge independently, then times the same
combined core path for composition checks.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton

from bench_e10_grouping import CFG, cache, error, write_csv
from bench_gpu import cuda_ms
from ofd_kernel.runtime.launcher import allocate_workspace, next_power_of_2
from ofd_kernel.runtime.subgrouped_launcher import subgrouped_paged_decode_core
from ofd_kernel.triton.kernels import merge_splitkv_kernel
from ofd_kernel.triton.subgrouped_kernels import subgrouped_partials_kernel

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "kernel/experiments/E13_phase_attribution/protocol.json"


def _rep_tensors(rep, k, v, kp, ks, vp, vs):
    if rep == "INT4":
        return kp, ks, vp, vs, 32
    if rep == "FP16":
        # Scale pointers are unused when IS_INT4=False. Match the frozen launcher
        # contract by passing resident tensors rather than allocating dummies.
        return k, k, v, v, 32
    raise ValueError(rep)


def launch_stage1(q, k, v, kp, ks, vp, vs, bt, lens, ws, *, rep, g, max_seq_len):
    b, hq, d = q.shape
    hkv = 4
    kx, ksx, vx, vsx, group_size = _rep_tensors(rep, k, v, kp, ks, vp, vs)
    ns = int(ws.partial_m.shape[2])
    expected_ns = math.ceil(max_seq_len / CFG.split_size)
    if ns != expected_ns:
        raise RuntimeError(f"workspace splits {ns} != expected {expected_ns}")

    partial_m, partial_l, partial_o = ws.partial_m, ws.partial_l, ws.partial_o
    grid = (b, hkv * ((hq // hkv) // g), ns)
    subgrouped_partials_kernel[grid](
        q, kx, ksx, vx, vsx, bt, lens,
        partial_m, partial_l, partial_o,
        stride_q_b=q.stride(0), stride_q_h=q.stride(1),
        stride_k_page=kx.stride(0), stride_k_head=kx.stride(1),
        stride_k_tok=kx.stride(2), stride_k_group=kx.stride(3) if rep == "INT4" else 0,
        stride_ks_page=ksx.stride(0) if rep == "INT4" else 0,
        stride_ks_head=ksx.stride(1) if rep == "INT4" else 0,
        stride_ks_tok=ksx.stride(2) if rep == "INT4" else 0,
        stride_ks_group=ksx.stride(3) if rep == "INT4" else 0,
        stride_v_page=vx.stride(0), stride_v_head=vx.stride(1),
        stride_v_tok=vx.stride(2), stride_v_group=vx.stride(3) if rep == "INT4" else 0,
        stride_vs_page=vsx.stride(0) if rep == "INT4" else 0,
        stride_vs_head=vsx.stride(1) if rep == "INT4" else 0,
        stride_vs_tok=vsx.stride(2) if rep == "INT4" else 0,
        stride_vs_group=vsx.stride(3) if rep == "INT4" else 0,
        stride_bt_b=bt.stride(0), stride_bt_p=bt.stride(1),
        stride_pm_b=partial_m.stride(0), stride_pm_h=partial_m.stride(1),
        stride_pm_s=partial_m.stride(2),
        stride_po_b=partial_o.stride(0), stride_po_h=partial_o.stride(1),
        stride_po_s=partial_o.stride(2),
        NUM_Q_HEADS=hq, NUM_KV_HEADS=hkv, HEAD_DIM=d, GROUP_SIZE=group_size,
        PAGE_SIZE=16, SPLIT_SIZE=CFG.split_size,
        TOKENS_PER_BLOCK=CFG.tokens_per_block, NUM_SPLITS=ns,
        GROUP_Q=g, IS_INT4=rep == "INT4",
        num_warps=CFG.num_warps, num_stages=CFG.num_stages,
    )


def launch_merge(q, ws):
    b, hq, d = q.shape
    ns = int(ws.partial_m.shape[2])
    split_block = next_power_of_2(ns)
    partial_m, partial_l, partial_o, out = (
        ws.partial_m, ws.partial_l, ws.partial_o, ws.out
    )
    merge_splitkv_kernel[(b, hq)](
        partial_m, partial_l, partial_o, out,
        stride_pm_b=partial_m.stride(0), stride_pm_h=partial_m.stride(1),
        stride_pm_s=partial_m.stride(2),
        stride_po_b=partial_o.stride(0), stride_po_h=partial_o.stride(1),
        stride_po_s=partial_o.stride(2),
        stride_o_b=out.stride(0), stride_o_h=out.stride(1),
        HEAD_DIM=d, NUM_SPLITS=ns, SPLIT_BLOCK=split_block,
        num_warps=4, num_stages=1,
    )
    return out


def run_core(q, k, v, kp, ks, vp, vs, bt, lens, ws, *, rep, g, max_seq_len):
    kx, ksx, vx, vsx, _ = _rep_tensors(rep, k, v, kp, ks, vp, vs)
    return subgrouped_paged_decode_core(
        q, kx, ksx, vx, vsx, bt, lens,
        config=CFG, workspace=ws, max_seq_len=max_seq_len,
        representation=rep, group_q=g,
    )


def explicit_core(q, k, v, kp, ks, vp, vs, bt, lens, ws, *, rep, g, max_seq_len):
    launch_stage1(q, k, v, kp, ks, vp, vs, bt, lens, ws,
                  rep=rep, g=g, max_seq_len=max_seq_len)
    return launch_merge(q, ws)


def worker(out: Path, process_id: int, smoke: bool):
    protocol = json.loads(PROTOCOL.read_text())
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    conditions = protocol["conditions"][:1] if smoke else protocol["conditions"]
    raw, checks = [], []

    order_conditions = list(conditions)
    random.Random(1313 + process_id).shuffle(order_conditions)

    for cond in order_conditions:
        t, b, label = cond["context"], cond["batch"], cond["label"]
        seed = 1313 + process_id * 100000 + t + b
        print("CONDITION", process_id, label, t, b, flush=True)
        q, k, v, kp, ks, vp, vs, bt, lens = cache(t, b, seed)
        ws = allocate_workspace(q, lens, CFG, max_seq_len=t)
        ns = int(ws.partial_m.shape[2])

        for rep in protocol["representations"]:
            for g in protocol["group_q"]:
                y_exp = explicit_core(q, k, v, kp, ks, vp, vs, bt, lens, ws,
                                      rep=rep, g=g, max_seq_len=t).clone()
                y_core = run_core(q, k, v, kp, ks, vp, vs, bt, lens, ws,
                                  rep=rep, g=g, max_seq_len=t).clone()
                rel = error(y_core, y_exp)
                finite = bool(torch.isfinite(y_exp).all().item() and torch.isfinite(y_core).all().item())
                checks.append({
                    "process_id": process_id,
                    "label": label,
                    "context": t,
                    "batch": b,
                    "representation": rep,
                    "group_q": g,
                    "relative_error": rel,
                    "finite": finite,
                    "git_sha": sha,
                })
                write_csv(out / f"correctness_p{process_id}.csv", checks)
                if (not finite) or rel > protocol["correctness_threshold_relative_l2"]:
                    raise RuntimeError(f"E13 explicit/core mismatch {rep} g{g}: {rel}")

        variants = [
            (rep, g, scope)
            for rep in protocol["representations"]
            for g in protocol["group_q"]
            for scope in protocol["timing_scopes"]
        ]

        for round_id in range(protocol["rounds"]):
            order = variants.copy()
            random.Random(seed + round_id * 1009).shuffle(order)
            for ordinal, (rep, g, scope) in enumerate(order):
                if scope == "stage1":
                    fn = lambda rep=rep, g=g: launch_stage1(
                        q, k, v, kp, ks, vp, vs, bt, lens, ws,
                        rep=rep, g=g, max_seq_len=t,
                    )
                elif scope == "merge":
                    # Populate correct partials outside the timed merge block.
                    launch_stage1(q, k, v, kp, ks, vp, vs, bt, lens, ws,
                                  rep=rep, g=g, max_seq_len=t)
                    torch.cuda.synchronize()
                    fn = lambda: launch_merge(q, ws)
                elif scope == "core":
                    fn = lambda rep=rep, g=g: run_core(
                        q, k, v, kp, ks, vp, vs, bt, lens, ws,
                        rep=rep, g=g, max_seq_len=t,
                    )
                else:
                    raise ValueError(scope)

                ms = cuda_ms(fn, protocol["warmup"], protocol["repeat"])
                raw.append({
                    "process_id": process_id,
                    "round_id": round_id,
                    "variant_order": ordinal,
                    "label": label,
                    "context": t,
                    "batch": b,
                    "representation": rep,
                    "group_q": g,
                    "timing_scope": scope,
                    "latency_ms": ms,
                    "num_splits": ns,
                    "stage1_program_count": b * 4 * (6 // g) * ns,
                    "Hq": 24,
                    "Hkv": 4,
                    "head_dim": 256,
                    "split_size": CFG.split_size,
                    "tokens_per_block": CFG.tokens_per_block,
                    "num_warps": CFG.num_warps,
                    "num_stages": CFG.num_stages,
                    "page_size": 16,
                    "quant_group_size": 32,
                    "seed": seed,
                    "warmup": protocol["warmup"],
                    "repeat": protocol["repeat"],
                    "git_sha": sha,
                    "GPU": torch.cuda.get_device_name(),
                    "CUDA": torch.version.cuda,
                    "PyTorch": torch.__version__,
                    "Triton": triton.__version__,
                })
                write_csv(out / f"raw_p{process_id}.csv", raw)

        del ws, q, k, v, kp, ks, vp, vs, bt, lens
        gc.collect()
        torch.cuda.empty_cache()


def _merge_csv(out: Path, pattern: str, name: str):
    rows = []
    for f in sorted(out.glob(pattern)):
        rows.extend(csv.DictReader(f.open()))
    write_csv(out / name, rows)
    return rows


def _composition(raw_rows, out: Path):
    keyed = {}
    for r in raw_rows:
        key = (
            r["process_id"], r["round_id"], r["label"], r["context"],
            r["batch"], r["representation"], r["group_q"],
        )
        keyed.setdefault(key, {})[r["timing_scope"]] = float(r["latency_ms"])

    rows = []
    for key, vals in keyed.items():
        if not {"core", "stage1", "merge"}.issubset(vals):
            continue
        core, s1, merge = vals["core"], vals["stage1"], vals["merge"]
        denom = s1 + merge
        rows.append({
            "process_id": key[0],
            "round_id": key[1],
            "label": key[2],
            "context": key[3],
            "batch": key[4],
            "representation": key[5],
            "group_q": key[6],
            "core_ms": core,
            "stage1_ms": s1,
            "merge_ms": merge,
            "stage1_plus_merge_ms": denom,
            "composition_residual_ms": core - denom,
            "composition_ratio_sum_over_core": denom / core if core else float("nan"),
            "stage1_fraction_of_sum": s1 / denom if denom else float("nan"),
            "merge_fraction_of_sum": merge / denom if denom else float("nan"),
        })
    write_csv(out / "composition.csv", rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path)
    p.add_argument("--worker", type=int)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()

    if a.worker is not None:
        worker(a.out, a.worker, a.smoke)
        return

    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty:
        raise RuntimeError("Tracked worktree must be clean for E13")

    protocol = json.loads(PROTOCOL.read_text())
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = a.out or ROOT / "kernel/results/E13_phase_attribution" / f"{stamp}_{os.getpid()}"
    out.mkdir(parents=True, exist_ok=False)

    processes = 1 if a.smoke else protocol["processes"]
    manifest = {
        "experiment": "E13_phase_attribution",
        "git_sha": sha,
        "git_dirty": False,
        "smoke": a.smoke,
        "protocol": protocol,
        "status": "RUNNING",
        "claim_eligible": False,
        "created_utc": stamp,
        "commands": [],
        "note": "Frozen subgrouped Stage-1 and existing merge; no hot-kernel modification.",
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("OUT", out, flush=True)

    rc = 0
    with (out / "run.log").open("w") as log:
        for pid in range(processes):
            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(pid), "--out", str(out)]
            if a.smoke:
                cmd.append("--smoke")
            manifest["commands"].append(cmd)
            (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
            rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
            if rc:
                break

    raw = _merge_csv(out, "raw_p*.csv", "raw.csv")
    _merge_csv(out, "correctness_p*.csv", "correctness.csv")
    _composition(raw, out)

    manifest.update(status="COMPLETED" if rc == 0 else "FAILED", return_code=rc)
    manifest["claim_eligible"] = rc == 0 and not a.smoke
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest["status"], out, flush=True)
    if rc:
        raise RuntimeError("E13 failed; partial evidence retained")


if __name__ == "__main__":
    main()
