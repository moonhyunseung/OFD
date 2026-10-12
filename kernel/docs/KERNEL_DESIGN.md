# Kernel design

## Current correctness-first Stage 1

Grid: `(request, q_head, split)`.

Each program traverses one context split, resolves logical pages through the block table, directly reads packed INT4 K/V, dequantizes in registers, and accumulates a numerically stable local softmax state.

## Stage 2

A second kernel merges split states `(m,l,o)` into the final output.

## Known non-optimized parts

1. Current q-head grid rereads shared KV under GQA; grouped KV reuse is intentionally deferred to E06.
2. The correctness draft may load a packed byte redundantly for its low/high nibbles; vectorized byte→two-value unpack is an optimization task after E02.
3. Query/output FWHT currently crosses the PyTorch reference boundary; dedicated/fused GPU FWHT is a later optimization.
4. No TMA, warp specialization, persistent scheduling, or architecture-specific SASS claim is present yet.

These are explicit TODOs, not hidden limitations.

## v2.3 measurement boundary

The runtime exposes two conceptual levels:

- `triton_paged_decode_core(...)`: branch-free packed-INT4 Stage 1 + Stage 2 only. This is the claim-eligible core timing path used by E04a/E07.
- `triton_paged_decode(...)`: correctness wrapper that also applies the current PyTorch FWHT reference at Q/output boundaries. It is not eligible for end-to-end latency claims until a production GPU FWHT replaces the reference transform.

A preallocated workspace carries host-known `max_seq_len` and split count. Timed core execution performs no CUDA-to-CPU `.item()` synchronization. Expensive block-table/page-index validation is opt-in debug logic used by correctness runs only.
