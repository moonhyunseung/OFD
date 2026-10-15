
import argparse, csv, gc, math, os, statistics, time
from pathlib import Path

import torch
try:
    import triton
    import triton.language as tl
except Exception as e:
    raise RuntimeError(
        "Triton is required. Check `python3 -c \"import triton; print(triton.__version__)\"`."
    ) from e

# ============================================================
# CONFIG
# ============================================================
DTYPE = torch.bfloat16
HEADS = 32
HEAD_DIM = 128
PAGE_SIZE = 16
GROUP_SIZE = 32
NUM_GROUPS = HEAD_DIM // GROUP_SIZE
SEED = 1337

FULL_CONTEXTS = [8_192, 16_384, 32_768, 65_536]
FULL_BATCHES = [1, 2, 4, 8, 16, 32]

SMOKE_CONTEXTS = [8_192, 32_768]
SMOKE_BATCHES = [1, 4]

assert HEAD_DIM % GROUP_SIZE == 0
assert PAGE_SIZE & (PAGE_SIZE - 1) == 0
assert HEAD_DIM & (HEAD_DIM - 1) == 0


# ============================================================
# TRITON PAGED ATTENTION
# Same page table / online-softmax structure for both variants.
# ============================================================

@triton.jit
def paged_bf16_decode_kernel(
    Q, K, V, BLOCK_TABLE, SEQ_LENS, OUT,
    bt_stride,
    H: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SM_SCALE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    d = tl.arange(0, BLOCK_D)
    dm = d < D
    t = tl.arange(0, P)

    q_base = (b * H + h) * D
    q = tl.load(Q + q_base + d, mask=dm, other=0.0).to(tl.float32)

    seq_len = tl.load(SEQ_LENS + b).to(tl.int32)
    n_pages = (seq_len + P - 1) // P

    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for lp in tl.range(0, n_pages):
        phys = tl.load(BLOCK_TABLE + b * bt_stride + lp).to(tl.int64)

        pos = lp * P + t
        valid_t = pos < seq_len

        # cache: [num_pages, P, H, D], contiguous
        base = ((phys * P + t[:, None]) * H + h) * D
        addr = base + d[None, :]

        mask = valid_t[:, None] & dm[None, :]
        k = tl.load(K + addr, mask=mask, other=0.0).to(tl.float32)

        score = tl.sum(k * q[None, :], axis=1) * SM_SCALE
        score = tl.where(valid_t, score, -float("inf"))

        m_new = tl.maximum(m_prev, tl.max(score, axis=0))
        alpha = tl.exp(m_prev - m_new)
        p = tl.exp(score - m_new)

        v = tl.load(V + addr, mask=mask, other=0.0).to(tl.float32)

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_prev = l_prev * alpha + tl.sum(p, axis=0)
        m_prev = m_new

    out_base = (b * H + h) * D
    tl.store(OUT + out_base + d, (acc / l_prev).to(tl.bfloat16), mask=dm)


@triton.jit
def paged_ofd_int4_decode_kernel(
    Q_ROT,
    K_PACKED, K_SCALE,
    V_PACKED, V_SCALE,
    BLOCK_TABLE, SEQ_LENS,
    OUT_ROT,
    bt_stride,
    H: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,
    GROUP: tl.constexpr,
    N_GROUPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SM_SCALE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    d = tl.arange(0, BLOCK_D)
    dm = d < D
    t = tl.arange(0, P)

    q_base = (b * H + h) * D
    q = tl.load(Q_ROT + q_base + d, mask=dm, other=0.0).to(tl.float32)

    seq_len = tl.load(SEQ_LENS + b).to(tl.int32)
    n_pages = (seq_len + P - 1) // P

    byte_idx = d // 2
    nib_shift = (d & 1) * 4
    group_idx = d // GROUP

    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for lp in tl.range(0, n_pages):
        phys = tl.load(BLOCK_TABLE + b * bt_stride + lp).to(tl.int64)

        pos = lp * P + t
        valid_t = pos < seq_len
        mask = valid_t[:, None] & dm[None, :]

        # packed cache: [num_pages, P, H, D/2]
        pk_base = ((phys * P + t[:, None]) * H + h) * (D // 2)
        pk_raw = tl.load(
            K_PACKED + pk_base + byte_idx[None, :],
            mask=mask,
            other=0,
        ).to(tl.int32)

        k_q = ((pk_raw >> nib_shift[None, :]) & 0xF) - 8

        # scales: [num_pages, P, H, N_GROUPS]
        sk_base = ((phys * P + t[:, None]) * H + h) * N_GROUPS
        sk = tl.load(
            K_SCALE + sk_base + group_idx[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        k = k_q.to(tl.float32) * sk

        score = tl.sum(k * q[None, :], axis=1) * SM_SCALE
        score = tl.where(valid_t, score, -float("inf"))

        m_new = tl.maximum(m_prev, tl.max(score, axis=0))
        alpha = tl.exp(m_prev - m_new)
        p = tl.exp(score - m_new)

        pv_raw = tl.load(
            V_PACKED + pk_base + byte_idx[None, :],
            mask=mask,
            other=0,
        ).to(tl.int32)

        v_q = ((pv_raw >> nib_shift[None, :]) & 0xF) - 8

        sv = tl.load(
            V_SCALE + sk_base + group_idx[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        v = v_q.to(tl.float32) * sv

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_prev = l_prev * alpha + tl.sum(p, axis=0)
        m_prev = m_new

    out_base = (b * H + h) * D
    tl.store(OUT_ROT + out_base + d, (acc / l_prev).to(tl.bfloat16), mask=dm)


# ============================================================
# HOST UTILS
# ============================================================

def choose_gpu():
    stats = []
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            free, total = torch.cuda.mem_get_info()
        stats.append((free, total, i))
    free, total, idx = max(stats)
    torch.cuda.set_device(idx)
    return idx, free, total


def make_hadamard(device):
    h = torch.tensor([[1.0]], dtype=torch.float32)
    while h.shape[0] < HEAD_DIM:
        h = torch.cat(
            [torch.cat([h, h], dim=1),
             torch.cat([h, -h], dim=1)],
            dim=0,
        )
    h /= math.sqrt(HEAD_DIM)
    return h.to(device=device, dtype=DTYPE).contiguous()


def rotate(x, hmat):
    # row-vector convention: x' = x H; H is symmetric/self-inverse
    return torch.matmul(x, hmat)


def quantize_int4_groupwise(x):
    shape = x.shape
    g = shape[-1] // GROUP_SIZE
    xf = x.float().reshape(*shape[:-1], g, GROUP_SIZE)

    scale = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7.0
    q = torch.round(xf / scale).clamp(-8, 7).to(torch.int16)
    code = (q + 8).to(torch.uint8)

    lo = code[..., 0::2]
    hi = code[..., 1::2]
    packed = (lo | (hi << 4)).reshape(*shape[:-1], HEAD_DIM // 2).contiguous()

    return packed, scale.squeeze(-1).to(torch.float16).contiguous()


def dequantize_int4_groupwise(packed, scale):
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    code = torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], HEAD_DIM)
    q = code.to(torch.int16) - 8
    gshape = (*packed.shape[:-1], NUM_GROUPS, GROUP_SIZE)
    q = q.reshape(gshape).float()
    x = q * scale.float().unsqueeze(-1)
    return x.reshape(*packed.shape[:-1], HEAD_DIM).to(DTYPE)


def make_block_table(batch, context, device):
    n_pages = (context + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = batch * n_pages
    gen = torch.Generator(device="cpu").manual_seed(SEED + batch + context)
    perm = torch.randperm(total_pages, generator=gen, dtype=torch.int64)
    bt = perm.reshape(batch, n_pages).to(device=device, dtype=torch.int32)
    seq = torch.full((batch,), context, device=device, dtype=torch.int32)
    return bt.contiguous(), seq.contiguous(), total_pages


def bf16_cache_bytes(batch, context):
    pages = batch * ((context + PAGE_SIZE - 1) // PAGE_SIZE)
    return pages * PAGE_SIZE * HEADS * HEAD_DIM * 2 * 2  # K+V, bf16


def ofd_cache_bytes(batch, context):
    pages = batch * ((context + PAGE_SIZE - 1) // PAGE_SIZE)
    values = pages * PAGE_SIZE * HEADS
    one = values * (HEAD_DIM // 2 + NUM_GROUPS * 2)  # packed + fp16 scales
    return 2 * one  # K+V


def gb(x):
    return x / (1024 ** 3)


def launch_bf16(q, k, v, bt, seq, out):
    grid = (q.shape[0], HEADS)
    paged_bf16_decode_kernel[grid](
        q, k, v, bt, seq, out,
        bt.stride(0),
        H=HEADS,
        D=HEAD_DIM,
        P=PAGE_SIZE,
        BLOCK_D=HEAD_DIM,
        SM_SCALE=1.0 / math.sqrt(HEAD_DIM),
        num_warps=4,
        num_stages=1,
    )


def launch_ofd(q_rot, pk, sk, pv, sv, bt, seq, out_rot):
    grid = (q_rot.shape[0], HEADS)
    paged_ofd_int4_decode_kernel[grid](
        q_rot, pk, sk, pv, sv, bt, seq, out_rot,
        bt.stride(0),
        H=HEADS,
        D=HEAD_DIM,
        P=PAGE_SIZE,
        GROUP=GROUP_SIZE,
        N_GROUPS=NUM_GROUPS,
        BLOCK_D=HEAD_DIM,
        SM_SCALE=1.0 / math.sqrt(HEAD_DIM),
        num_warps=4,
        num_stages=1,
    )


def bench(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times), min(times), max(times)


def free_all(*xs):
    del xs
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# ============================================================
# CORRECTNESS GATE
# ============================================================

@torch.inference_mode()
def correctness_gate(device, hmat):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    B = 1
    T = 512
    bt, seq, total_pages = make_block_table(B, T, device)
    n_pages = T // PAGE_SIZE
    phys = bt[0].long()

    q = torch.randn((B, HEADS, HEAD_DIM), device=device, dtype=DTYPE)
    k_log = torch.randn((B, T, HEADS, HEAD_DIM), device=device, dtype=DTYPE)
    v_log = torch.randn_like(k_log)

    # FP reference
    scores = torch.einsum(
        "bhd,bthd->bht", q.float(), k_log.float()
    ) / math.sqrt(HEAD_DIM)
    att = torch.softmax(scores, dim=-1)
    y_ref = torch.einsum("bht,bthd->bhd", att, v_log.float()).to(DTYPE)

    # map logical pages -> randomized physical pages
    k_pages = k_log.reshape(n_pages, PAGE_SIZE, HEADS, HEAD_DIM)
    v_pages = v_log.reshape(n_pages, PAGE_SIZE, HEADS, HEAD_DIM)

    k_bf = torch.empty(
        (total_pages, PAGE_SIZE, HEADS, HEAD_DIM),
        device=device, dtype=DTYPE
    )
    v_bf = torch.empty_like(k_bf)
    k_bf[phys] = k_pages
    v_bf[phys] = v_pages

    y_bf = torch.empty_like(q)
    launch_bf16(q, k_bf, v_bf, bt, seq, y_bf)
    torch.cuda.synchronize()

    # OFD cache from the SAME original K/V
    k_rot = rotate(k_pages, hmat)
    v_rot = rotate(v_pages, hmat)
    pk_log, sk_log = quantize_int4_groupwise(k_rot)
    pv_log, sv_log = quantize_int4_groupwise(v_rot)

    pk = torch.empty(
        (total_pages, PAGE_SIZE, HEADS, HEAD_DIM // 2),
        device=device, dtype=torch.uint8
    )
    pv = torch.empty_like(pk)
    sk = torch.empty(
        (total_pages, PAGE_SIZE, HEADS, NUM_GROUPS),
        device=device, dtype=torch.float16
    )
    sv = torch.empty_like(sk)

    pk[phys] = pk_log
    pv[phys] = pv_log
    sk[phys] = sk_log
    sv[phys] = sv_log

    q_rot = rotate(q, hmat).contiguous()
    y_rot = torch.empty_like(q)
    launch_ofd(q_rot, pk, sk, pv, sv, bt, seq, y_rot)
    torch.cuda.synchronize()
    y_ofd = rotate(y_rot, hmat)

    def rel(a, b):
        return ((a.float() - b.float()).norm() /
                (a.float().norm() + 1e-12)).item()

    bf_err = rel(y_ref, y_bf)
    ofd_err = rel(y_ref, y_ofd)

    print("\n=== CORRECTNESS GATE ===")
    print(f"BF16 paged vs torch reference : {bf_err:.6e}")
    print(f"OFD INT4 vs torch reference   : {ofd_err:.6e}")

    if bf_err > 5e-3:
        raise RuntimeError(f"BF16 kernel correctness failed: {bf_err}")

    if not math.isfinite(ofd_err) or ofd_err > 0.35:
        raise RuntimeError(f"OFD kernel correctness suspicious: {ofd_err}")

    free_all(
        bt, seq, q, k_log, v_log, y_ref,
        k_bf, v_bf, y_bf,
        k_rot, v_rot, pk_log, sk_log, pv_log, sv_log,
        pk, pv, sk, sv, q_rot, y_rot, y_ofd
    )


# ============================================================
# CACHE BUILDERS (not timed)
# ============================================================

@torch.inference_mode()
def build_bf16_cache(total_pages, device):
    shape = (total_pages, PAGE_SIZE, HEADS, HEAD_DIM)
    k = torch.empty(shape, device=device, dtype=DTYPE)
    v = torch.empty_like(k)
    k.normal_()
    v.normal_()
    return k, v


@torch.inference_mode()
def build_ofd_cache(total_pages, device, hmat, chunk_pages=128):
    pshape = (total_pages, PAGE_SIZE, HEADS, HEAD_DIM // 2)
    sshape = (total_pages, PAGE_SIZE, HEADS, NUM_GROUPS)

    pk = torch.empty(pshape, device=device, dtype=torch.uint8)
    pv = torch.empty_like(pk)
    sk = torch.empty(sshape, device=device, dtype=torch.float16)
    sv = torch.empty_like(sk)

    for start in range(0, total_pages, chunk_pages):
        end = min(start + chunk_pages, total_pages)
        n = end - start

        x = torch.randn(
            (n, PAGE_SIZE, HEADS, HEAD_DIM),
            device=device, dtype=DTYPE
        )
        xr = rotate(x, hmat)
        p, s = quantize_int4_groupwise(xr)
        pk[start:end].copy_(p)
        sk[start:end].copy_(s)
        del x, xr, p, s

        x = torch.randn(
            (n, PAGE_SIZE, HEADS, HEAD_DIM),
            device=device, dtype=DTYPE
        )
        xr = rotate(x, hmat)
        p, s = quantize_int4_groupwise(xr)
        pv[start:end].copy_(p)
        sv[start:end].copy_(s)
        del x, xr, p, s

    torch.cuda.synchronize()
    return pk, sk, pv, sv


# ============================================================
# ONE SWEEP POINT
# ============================================================

@torch.inference_mode()
def run_point(batch, context, device, hmat, warmup, repeat, mem_fraction):
    bt, seq, total_pages = make_block_table(batch, context, device)
    q = torch.randn((batch, HEADS, HEAD_DIM), device=device, dtype=DTYPE)

    # ---------------- BF16 ----------------
    gc.collect()
    torch.cuda.empty_cache()
    free0, _ = torch.cuda.mem_get_info()

    est_bf = bf16_cache_bytes(batch, context)
    if est_bf > free0 * mem_fraction:
        bf_ms = bf_min = bf_max = float("nan")
        bf_tps = float("nan")
        bf_status = "SKIP_MEM"
    else:
        k, v = build_bf16_cache(total_pages, device)
        out = torch.empty_like(q)

        def fn_bf():
            launch_bf16(q, k, v, bt, seq, out)

        bf_ms, bf_min, bf_max = bench(fn_bf, warmup, repeat)
        bf_tps = batch * 1000.0 / bf_ms
        bf_status = "OK"

        del k, v, out
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # ---------------- OFD ----------------
    free1, _ = torch.cuda.mem_get_info()
    est_ofd = ofd_cache_bytes(batch, context)

    if est_ofd > free1 * mem_fraction:
        ofd_kernel_ms = ofd_e2e_ms = float("nan")
        ofd_tps = speedup = float("nan")
        ofd_status = "SKIP_MEM"
    else:
        pk, sk, pv, sv = build_ofd_cache(total_pages, device, hmat)

        q_rot = torch.empty_like(q)
        out_rot = torch.empty_like(q)
        out_final = torch.empty_like(q)

        # dense H matmul is only a proxy for transform overhead.
        # kernel-only metric isolates the paged compressed-KV path.
        q2 = q.view(-1, HEAD_DIM)
        qr2 = q_rot.view(-1, HEAD_DIM)
        or2 = out_rot.view(-1, HEAD_DIM)
        of2 = out_final.view(-1, HEAD_DIM)

        torch.mm(q2, hmat, out=qr2)

        def fn_ofd_kernel():
            launch_ofd(q_rot, pk, sk, pv, sv, bt, seq, out_rot)

        ofd_kernel_ms, _, _ = bench(fn_ofd_kernel, warmup, repeat)

        def fn_ofd_e2e():
            torch.mm(q2, hmat, out=qr2)
            launch_ofd(q_rot, pk, sk, pv, sv, bt, seq, out_rot)
            torch.mm(or2, hmat, out=of2)

        ofd_e2e_ms, _, _ = bench(fn_ofd_e2e, warmup, repeat)

        ofd_tps = batch * 1000.0 / ofd_e2e_ms
        speedup = bf_ms / ofd_e2e_ms if math.isfinite(bf_ms) else float("nan")
        ofd_status = "OK"

        del pk, sk, pv, sv, q_rot, out_rot, out_final
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    row = {
        "context": context,
        "batch": batch,
        "bf16_status": bf_status,
        "ofd_status": ofd_status,
        "bf16_ms": bf_ms,
        "ofd_kernel_ms": ofd_kernel_ms,
        "ofd_e2e_ms": ofd_e2e_ms,
        "bf16_tok_s": bf_tps,
        "ofd_tok_s": ofd_tps,
        "bf16_over_ofd_e2e": speedup,
        "bf16_kv_gb": gb(est_bf),
        "ofd_kv_gb": gb(est_ofd),
        "compression": est_bf / est_ofd,
    }

    del bt, seq, q
    gc.collect()
    torch.cuda.empty_cache()
    return row


# ============================================================
# MAIN / RESUME / LOGGING
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--kv-budget-gb", type=float, default=40.0)
    ap.add_argument(
        "--mem-fraction", type=float, default=0.70,
        help="skip a single variant if estimated cache exceeds this fraction of current free VRAM"
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    idx, free, total = choose_gpu()
    device = torch.device(f"cuda:{idx}")

    print("torch :", torch.__version__)
    print("triton:", triton.__version__)
    print("cuda  :", torch.version.cuda)
    print("gpu   :", idx, torch.cuda.get_device_name(idx))
    print("free  :", f"{gb(free):.2f} / {gb(total):.2f} GiB")
    print("geom  :", f"H={HEADS}, D={HEAD_DIM}, page={PAGE_SIZE}, int4 group={GROUP_SIZE}")

    hmat = make_hadamard(device)
    correctness_gate(device, hmat)

    if args.smoke:
        contexts = SMOKE_CONTEXTS
        batches = SMOKE_BATCHES
        warmup, repeat = 1, 3
        tag = "smoke"
    else:
        contexts = FULL_CONTEXTS
        batches = FULL_BATCHES
        warmup, repeat = 3, 10
        tag = "full"

    out_csv = Path(f"paged_ofd_{tag}.csv")
    done = set()

    if args.resume and out_csv.exists():
        with out_csv.open() as f:
            for r in csv.DictReader(f):
                done.add((int(r["context"]), int(r["batch"])))
        print(f"resume: {len(done)} points already complete")

    fields = [
        "context", "batch",
        "bf16_status", "ofd_status",
        "bf16_ms", "ofd_kernel_ms", "ofd_e2e_ms",
        "bf16_tok_s", "ofd_tok_s", "bf16_over_ofd_e2e",
        "bf16_kv_gb", "ofd_kv_gb", "compression",
    ]

    if not out_csv.exists() or not args.resume:
        with out_csv.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    points = [(t, b) for t in contexts for b in batches if (t, b) not in done]
    print(f"\npoints: {len(points)} | CSV checkpoint: {out_csv}")

    t_start = time.time()
    completed = 0

    for context, batch in points:
        print("\n" + "=" * 76)
        print(f"context={context:,} batch={batch}")
        print("=" * 76)

        try:
            row = run_point(
                batch, context, device, hmat,
                warmup=warmup,
                repeat=repeat,
                mem_fraction=args.mem_fraction,
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            row = {
                "context": context, "batch": batch,
                "bf16_status": "OOM", "ofd_status": "OOM",
                "bf16_ms": float("nan"),
                "ofd_kernel_ms": float("nan"),
                "ofd_e2e_ms": float("nan"),
                "bf16_tok_s": float("nan"),
                "ofd_tok_s": float("nan"),
                "bf16_over_ofd_e2e": float("nan"),
                "bf16_kv_gb": gb(bf16_cache_bytes(batch, context)),
                "ofd_kv_gb": gb(ofd_cache_bytes(batch, context)),
                "compression": bf16_cache_bytes(batch, context) / ofd_cache_bytes(batch, context),
            }

        print(
            f"BF16: {row['bf16_ms']:.4f} ms | "
            f"{row['bf16_tok_s']:.1f} tok/s | "
            f"KV={row['bf16_kv_gb']:.3f} GiB"
        )
        print(
            f"OFD : kernel={row['ofd_kernel_ms']:.4f} ms | "
            f"e2e={row['ofd_e2e_ms']:.4f} ms | "
            f"{row['ofd_tok_s']:.1f} tok/s | "
            f"KV={row['ofd_kv_gb']:.3f} GiB"
        )
        print(
            f"BF16/OFD-e2e speedup={row['bf16_over_ofd_e2e']:.4f}x | "
            f"compression={row['compression']:.4f}x"
        )

        # capacity extrapolation to L layers under fixed KV budget
        per_seq_bf = bf16_cache_bytes(1, context) * args.layers
        per_seq_ofd = ofd_cache_bytes(1, context) * args.layers
        budget = args.kv_budget_gb * (1024 ** 3)

        cap_bf = int(budget // per_seq_bf)
        cap_ofd = int(budget // per_seq_ofd)

        print(
            f"capacity @ {args.layers} layers / {args.kv_budget_gb:.1f} GiB KV budget: "
            f"BF16={cap_bf} seq | OFD={cap_ofd} seq"
        )

        with out_csv.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writerow(row)
            f.flush()
            os.fsync(f.fileno())

        completed += 1
        elapsed = time.time() - t_start
        avg = elapsed / completed
        remaining = avg * (len(points) - completed)
        print(f"progress {completed}/{len(points)} | ETA ~ {remaining/60:.1f} min")

    # summary: best throughput for each context among measured batch sizes
    rows = []
    with out_csv.open() as f:
        for r in csv.DictReader(f):
            for k in [
                "bf16_ms", "ofd_kernel_ms", "ofd_e2e_ms",
                "bf16_tok_s", "ofd_tok_s", "bf16_over_ofd_e2e",
                "bf16_kv_gb", "ofd_kv_gb", "compression"
            ]:
                r[k] = float(r[k])
            r["context"] = int(r["context"])
            r["batch"] = int(r["batch"])
            rows.append(r)

    print("\n" + "=" * 76)
    print("BEST MEASURED THROUGHPUT PER CONTEXT")
    print("=" * 76)
    for t in contexts:
        rr = [r for r in rows if r["context"] == t]
        rr_bf = [r for r in rr if math.isfinite(r["bf16_tok_s"])]
        rr_ofd = [r for r in rr if math.isfinite(r["ofd_tok_s"])]

        best_bf = max(rr_bf, key=lambda x: x["bf16_tok_s"]) if rr_bf else None
        best_ofd = max(rr_ofd, key=lambda x: x["ofd_tok_s"]) if rr_ofd else None

        print(f"T={t:,}")
        if best_bf:
            print(
                f"  BF16 best: B={best_bf['batch']:>2} "
                f"{best_bf['bf16_tok_s']:.1f} tok/s"
            )
        if best_ofd:
            print(
                f"  OFD  best: B={best_ofd['batch']:>2} "
                f"{best_ofd['ofd_tok_s']:.1f} tok/s"
            )

    print("\nCSV:", out_csv)
    print("Interpretation:")
    print("- bf16_over_ofd_e2e > 1: OFD is faster at the SAME (T,B).")
    print("- Even when < 1, compare max feasible B under the fixed KV budget.")
    print("- ofd_kernel_ms isolates the compressed paged-KV path.")
    print("- ofd_e2e_ms includes dense-H query/output transforms; replace with fused FWHT later.")


if __name__ == "__main__":
    main()
