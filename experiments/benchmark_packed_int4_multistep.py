import math, os, time, urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Config
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

geom = {
    "name": "7B-class",
    "vocab_size": 32000,
    "n_layer": 32,
    "n_head": 32,
    "n_embd": 4096,
}

B = 1 if device == "cuda" else 1
T_cache = 8192
decode_steps = 128
group_size = 32
repeat = 3
warmup = 1

torch.manual_seed(1337)
if device == "cuda":
    torch.cuda.manual_seed_all(1337)

vocab_size = geom["vocab_size"]
n_head = geom["n_head"]
n_embd = geom["n_embd"]
D = n_embd // n_head

assert D % group_size == 0
assert (D & (D - 1)) == 0, f"D={D} must be power of 2 for Hadamard"

print(f"device={device}, dtype={dtype}")
print(f"geometry={geom['name']} | vocab={vocab_size}, heads={n_head}, embd={n_embd}, D={D}")
print(f"B={B}, T_cache={T_cache}, decode_steps={decode_steps}, group_size={group_size}")

# -----------------------------
# Utils
# -----------------------------
def fmt_bytes(n):
    units = ["B","KB","MB","GB","TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0

def rel_fro_error(a, b):
    a = a.float()
    b = b.float()
    return ((a - b).norm() / (a.norm() + 1e-12)).item()

def cuda_sync():
    if device == "cuda":
        torch.cuda.synchronize()

def benchmark(fn, repeat=3, warmup=1):
    for _ in range(warmup):
        fn()
    cuda_sync()
    t0 = time.perf_counter()
    out = None
    for _ in range(repeat):
        out = fn()
    cuda_sync()
    t1 = time.perf_counter()
    return out, (t1 - t0) / repeat

def reset_peak():
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

def peak_alloc():
    if device != "cuda":
        return None
    return torch.cuda.max_memory_allocated()

# -----------------------------
# Hadamard
# -----------------------------
def hadamard_lastdim(x):
    Dloc = x.shape[-1]
    y = x.float()
    h = 1
    while h < Dloc:
        y = y.reshape(*y.shape[:-1], Dloc // (2 * h), 2, h)
        a = y[..., :, 0, :]
        b = y[..., :, 1, :]
        y = torch.cat([a + b, a - b], dim=-1)
        y = y.reshape(*x.shape[:-1], Dloc)
        h *= 2
    return (y / math.sqrt(Dloc)).to(x.dtype)

# -----------------------------
# Packed int4 quantization
# -----------------------------
def quantize_int4_packed_groupwise(x, group_size=32):
    assert x.shape[-1] % group_size == 0
    orig_shape = x.shape
    Dloc = orig_shape[-1]
    G = Dloc // group_size

    xf = x.float().reshape(-1, G, group_size)
    scale = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7.0
    q = torch.round(xf / scale).clamp(-8, 7).to(torch.int16)
    code = (q + 8).to(torch.uint8)

    lo = code[..., 0::2]
    hi = code[..., 1::2]
    packed = (lo | (hi << 4)).contiguous()

    return {
        "packed": packed,
        "scale": scale.squeeze(-1).to(torch.float16),
        "orig_shape": orig_shape,
        "group_size": group_size,
    }

def dequantize_int4_packed_groupwise(qobj, out_dtype=torch.float32):
    packed = qobj["packed"]
    scale = qobj["scale"]
    orig_shape = qobj["orig_shape"]
    group_size = qobj["group_size"]

    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    code = torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], group_size)
    q = code.to(torch.int16) - 8
    x = q.float() * scale.float().unsqueeze(-1)
    x = x.reshape(orig_shape)
    return x.to(device=device, dtype=out_dtype)

def storage_bytes_quant(qobj):
    return qobj["packed"].numel() * qobj["packed"].element_size() + qobj["scale"].numel() * qobj["scale"].element_size()

# -----------------------------
# Attention
# -----------------------------
def attention_decode(q, k, v):
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])  # (B,H,1,T)
    att = F.softmax(scores.float(), dim=-1).to(q.dtype)
    y = att @ v
    return y

# -----------------------------
# Token source
# -----------------------------
path = "tiny_shakespeare.txt"
if not os.path.exists(path):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, path)

with open(path, "rb") as f:
    raw = f.read()
all_tokens = torch.tensor(list(raw), dtype=torch.long)

need = B * (T_cache + decode_steps)
if len(all_tokens) < need:
    raise RuntimeError("tiny_shakespeare.txt too short; reduce T_cache or decode_steps.")

xs = []
stride = max(1, (len(all_tokens) - (T_cache + decode_steps)) // B)
offset = 0
for _ in range(B):
    xs.append(all_tokens[offset:offset + T_cache + decode_steps])
    offset += stride
x = torch.stack(xs, dim=0).to(device)

# -----------------------------
# Minimal projection stack
# -----------------------------
emb = nn.Embedding(vocab_size, n_embd).to(device=device, dtype=dtype)
Wq = nn.Linear(n_embd, n_embd, bias=False).to(device=device, dtype=dtype)
Wk = nn.Linear(n_embd, n_embd, bias=False).to(device=device, dtype=dtype)
Wv = nn.Linear(n_embd, n_embd, bias=False).to(device=device, dtype=dtype)

with torch.no_grad():
    h = emb(x)
    h_cache = h[:, :T_cache, :]
    h_dec = h[:, T_cache:T_cache + decode_steps, :]

    k0 = Wk(h_cache).view(B, T_cache, n_head, D).transpose(1, 2).contiguous()
    v0 = Wv(h_cache).view(B, T_cache, n_head, D).transpose(1, 2).contiguous()
    q_steps = Wq(h_dec).view(B, decode_steps, n_head, D).transpose(1, 2).contiguous()
    k_steps = Wk(h_dec).view(B, decode_steps, n_head, D).transpose(1, 2).contiguous()
    v_steps = Wv(h_dec).view(B, decode_steps, n_head, D).transpose(1, 2).contiguous()

with torch.no_grad():
    k0_rot = hadamard_lastdim(k0)
    v0_rot = hadamard_lastdim(v0)
    qk_cache = quantize_int4_packed_groupwise(k0_rot, group_size=group_size)
    qv_cache = quantize_int4_packed_groupwise(v0_rot, group_size=group_size)

# -----------------------------
# Single-step sanity check
# -----------------------------
with torch.no_grad():
    q1 = q_steps[:, :, 0:1, :]
    k1 = k_steps[:, :, 0:1, :]
    v1 = v_steps[:, :, 0:1, :]

    k_fp = torch.cat([k0, k1], dim=2)
    v_fp = torch.cat([v0, v1], dim=2)
    y_fp = attention_decode(q1, k_fp, v_fp)

    k_rot_cache_dq = dequantize_int4_packed_groupwise(qk_cache, out_dtype=dtype)
    v_rot_cache_dq = dequantize_int4_packed_groupwise(qv_cache, out_dtype=dtype)

    k_naive = torch.cat([hadamard_lastdim(k_rot_cache_dq), k1], dim=2)
    v_naive = torch.cat([hadamard_lastdim(v_rot_cache_dq), v1], dim=2)
    y_naive = attention_decode(q1, k_naive, v_naive)

    q1_rot = hadamard_lastdim(q1)
    k1_rot = hadamard_lastdim(k1)
    v1_rot = hadamard_lastdim(v1)
    k_reassoc = torch.cat([k_rot_cache_dq, k1_rot], dim=2)
    v_reassoc = torch.cat([v_rot_cache_dq, v1_rot], dim=2)
    y_reassoc = hadamard_lastdim(attention_decode(q1_rot, k_reassoc, v_reassoc))

print("\n=== Single-step correctness ===")
print(f"FP vs naive-TQ rel error            : {rel_fro_error(y_fp, y_naive):.6e}")
print(f"FP vs order-flip-TQ rel error       : {rel_fro_error(y_fp, y_reassoc):.6e}")
print(f"naive-TQ vs order-flip-TQ rel error : {rel_fro_error(y_naive, y_reassoc):.6e}")

# -----------------------------
# Multi-step decode
# -----------------------------
def run_fp_multistep():
    with torch.no_grad():
        k_cur = k0
        v_cur = v0
        outs = []
        for s in range(decode_steps):
            q = q_steps[:, :, s:s+1, :]
            k_new = k_steps[:, :, s:s+1, :]
            v_new = v_steps[:, :, s:s+1, :]
            k_cur = torch.cat([k_cur, k_new], dim=2)
            v_cur = torch.cat([v_cur, v_new], dim=2)
            outs.append(attention_decode(q, k_cur, v_cur))
        return torch.cat(outs, dim=2)

def run_tq_naive_multistep():
    with torch.no_grad():
        qk_cur = qk_cache
        qv_cur = qv_cache
        outs = []
        for s in range(decode_steps):
            q = q_steps[:, :, s:s+1, :]
            k_new = k_steps[:, :, s:s+1, :]
            v_new = v_steps[:, :, s:s+1, :]

            k_rot_dq = dequantize_int4_packed_groupwise(qk_cur, out_dtype=dtype)
            v_rot_dq = dequantize_int4_packed_groupwise(qv_cur, out_dtype=dtype)
            k_cur = hadamard_lastdim(k_rot_dq)
            v_cur = hadamard_lastdim(v_rot_dq)

            k_cur = torch.cat([k_cur, k_new], dim=2)
            v_cur = torch.cat([v_cur, v_new], dim=2)
            outs.append(attention_decode(q, k_cur, v_cur))

            qk_cur = quantize_int4_packed_groupwise(hadamard_lastdim(k_cur), group_size=group_size)
            qv_cur = quantize_int4_packed_groupwise(hadamard_lastdim(v_cur), group_size=group_size)

        return torch.cat(outs, dim=2)

def run_tq_orderflip_multistep():
    with torch.no_grad():
        qk_cur = qk_cache
        qv_cur = qv_cache
        outs = []
        for s in range(decode_steps):
            q = q_steps[:, :, s:s+1, :]
            k_new = k_steps[:, :, s:s+1, :]
            v_new = v_steps[:, :, s:s+1, :]

            k_rot_dq = dequantize_int4_packed_groupwise(qk_cur, out_dtype=dtype)
            v_rot_dq = dequantize_int4_packed_groupwise(qv_cur, out_dtype=dtype)

            q_rot = hadamard_lastdim(q)
            k_new_rot = hadamard_lastdim(k_new)
            v_new_rot = hadamard_lastdim(v_new)

            k_rot_cur = torch.cat([k_rot_dq, k_new_rot], dim=2)
            v_rot_cur = torch.cat([v_rot_dq, v_new_rot], dim=2)

            y_rot = attention_decode(q_rot, k_rot_cur, v_rot_cur)
            outs.append(hadamard_lastdim(y_rot))

            qk_cur = quantize_int4_packed_groupwise(k_rot_cur, group_size=group_size)
            qv_cur = quantize_int4_packed_groupwise(v_rot_cur, group_size=group_size)

        return torch.cat(outs, dim=2)

reset_peak()
y_fp_multi, t_fp = benchmark(run_fp_multistep, repeat=repeat, warmup=warmup)
peak_fp = peak_alloc()

reset_peak()
y_naive_multi, t_naive = benchmark(run_tq_naive_multistep, repeat=repeat, warmup=warmup)
peak_naive = peak_alloc()

reset_peak()
y_reassoc_multi, t_reassoc = benchmark(run_tq_orderflip_multistep, repeat=repeat, warmup=warmup)
peak_reassoc = peak_alloc()

fp_kv_bytes = k0.numel() * k0.element_size() + v0.numel() * v0.element_size()
tq_kv_bytes = storage_bytes_quant(qk_cache) + storage_bytes_quant(qv_cache)

print("\n=== Multi-step decode correctness ===")
print(f"FP vs naive-TQ rel error            : {rel_fro_error(y_fp_multi, y_naive_multi):.6e}")
print(f"FP vs order-flip-TQ rel error       : {rel_fro_error(y_fp_multi, y_reassoc_multi):.6e}")
print(f"naive-TQ vs order-flip-TQ rel error : {rel_fro_error(y_naive_multi, y_reassoc_multi):.6e}")

print("\n=== Multi-step decode performance ===")
print(f"decode steps                        : {decode_steps}")
print(f"FP baseline total                   : {t_fp*1000:.3f} ms")
print(f"TQ naive total                      : {t_naive*1000:.3f} ms")
print(f"TQ order-flip total                 : {t_reassoc*1000:.3f} ms")
print(f"FP baseline ms/token                : {(t_fp*1000)/decode_steps:.4f}")
print(f"TQ naive ms/token                   : {(t_naive*1000)/decode_steps:.4f}")
print(f"TQ order-flip ms/token              : {(t_reassoc*1000)/decode_steps:.4f}")
print(f"TQ naive tok/s                      : {decode_steps/t_naive:.2f}")
print(f"TQ order-flip tok/s                 : {decode_steps/t_reassoc:.2f}")
print(f"speedup naive/order-flip            : {t_naive/t_reassoc:.2f}x")

print("\n=== Peak VRAM ===")
if device == "cuda":
    print(f"FP baseline peak allocated          : {fmt_bytes(peak_fp)}")
    print(f"TQ naive peak allocated             : {fmt_bytes(peak_naive)}")
    print(f"TQ order-flip peak allocated        : {fmt_bytes(peak_reassoc)}")
    print(f"peak ratio naive/order-flip         : {peak_naive/peak_reassoc:.2f}x")
else:
    print("CUDA not available")

print("\n=== KV cache storage ===")
print(f"FP KV bytes                         : {fmt_bytes(fp_kv_bytes)}")
print(f"packed-int4 rotated KV bytes        : {fmt_bytes(tq_kv_bytes)}")
print(f"compression vs FP                   : {fp_kv_bytes / tq_kv_bytes:.2f}x")
print(f"effective bits/value                : {4 + 16/group_size:.2f}")

print("\n=== Honest interpretation ===")
print("- This is a stronger prototype: packed int4 + long-context + multi-step decode.")
print("- It is NOT a full TurboQuant reimplementation.")
print("- Residual 1-bit QJL correction is not included.")
print("- Fused custom kernels are not included.")
print("- If order-flip stays faster with small extra error, that supports it as a TurboQuant-compatible decode-path optimization.")