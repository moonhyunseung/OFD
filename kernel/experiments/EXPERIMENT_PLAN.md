# OFD Triton v2 — Experiment Plan

## Research objective

Determine whether a direct packed-INT4 paged attention path can retain OFD's representation flexibility while approaching Plain-INT4 decode cost on Blackwell, without hiding costs through dense KV materialization.

## Primary questions

1. Is the paged Triton implementation numerically equivalent to the frozen CPU paged oracle within the predefined tolerance?
2. Does split-KV parallelism remove the high-concurrency collapse observed in the v1 prototype?
3. What runtime overhead does P/K/V/KV representation choice add when storage/layout/kernel architecture are held fixed?
4. Under equal KV-memory budget, does compression improve aggregate throughput relative to a matched FP16 paged baseline?
5. Does the result persist as context grows from 8K to 128K+ and under GQA?
6. Which Blackwell-specific optimizations are justified by profiler evidence rather than assumed in advance?

## Gate order

```text
E00 CPU reference
  ↓
E01 GPU compile smoke
  ↓
E02 numerical correctness
  ├─→ E04a core attention parity ─→ E07 core long-context curve
  │       └─→ E04b end-to-end OFD (blocked until GPU FWHT)
  ├─→ E03 split scaling
  └─→ E06 GQA reuse
          ↓
E05 same-memory-budget (after matched FP16 baseline exists)
          ↓
E08 Blackwell tuning
          ↓
E09 real-model serving
```

A downstream experiment must not be used for claims if its prerequisites are blocked or failed.

## Frozen common contract

- group size = 32
- page size = 16 unless explicitly varied
- signed INT4 storage, encoded as `q+8`, low nibble first
- symmetric `absmax/7` scale; normal finite quantization uses `[-7,7]`
- scale dtype = FP16
- FP32 online-softmax / split merge accumulators
- randomized logical→physical page mapping for correctness
- no dense dequantized KV materialization in the Triton target path

## Claim discipline

- E01: compile/execution only; never quote performance.
- E02: implementation correctness only; not model quality.
- E04a: core attention parity only; Q/output transforms excluded from timing.
- E04b: end-to-end OFD-vs-Plain overhead only after production GPU FWHT exists.
- E05: infrastructure/capacity claim only after matched FP16 paged baseline.
- E09: model-quality or serving claims only after held-out real-model evaluation.

## Required run artifacts

Every run gets a unique directory containing `MANIFEST.json`, raw results, and exact command lines. A result without a manifest is not eligible for paper/company claims.

## Provenance rule

E03 and all claim-eligible experiments after correctness require a clean Git worktree. Manifests record `git_dirty`, the exact porcelain status, and separate `execution_status` from `gate_status`. Observational experiments use `gate_status=NOT_APPLICABLE`.
