# E03_split_scaling — Split-KV Scaling

**Status:** `READY_AFTER_E02`

## Hypothesis

Test whether split-KV removes v1 high-concurrency collapse.

## Dependencies

E02_correctness

## Frozen controls

- Quantizer: symmetric groupwise INT4, `scale = absmax / 7`
- Encoding: `q + 8`, low nibble first
- Group size: 32
- Scale dtype: FP16
- Page size: 16 unless this experiment explicitly varies it
- Accumulation/reference merge: FP32
- Physical page placement: randomized for correctness tests
- P/K/V/KV representation semantics are fixed by `docs/MATH.md`

## Primary pass/fail gate

The archived v1 kernel is `kernel/src/ofd_kernel/historical_v1/benchmark_paged_ofd_triton.py`, copied verbatim from the user's historical `benchmark_paged_ofd_triton.py` (SHA256 `d453145048471ea2026b6370ef545ee9fc3a5c7fea947c60bf34c4e27b627303`). The original v1 kernel traverses physical pages serially in one program per request/head. E03 converts the same packed bytes and FP16 scales into its historical memory layout outside timing, uses shared BF16 query/output dtype, and compares against v2 split-KV core. Both read identical randomized page tables and packed values. No FWHT is timed. Shared-cache output relative error must be <= 0.01 before performance timing. This is a cross-implementation comparison, not a claim that the original v1 harness used the same dtype and geometry in every historical run.

Frozen matrix: context 8K/32K/64K/128K, batch 1/4/8/16/32, Q/KV=1, Hkv=8, D=128, page=16, group=32. Ratios are observational; do not change the matrix because of measured performance.

## Outputs

Every actual run writes to `kernel/results/E03_split_scaling/<run_id>/` and must contain:

- `MANIFEST.json`
- raw CSV/logs
- processed summaries only if derived from that run

## Change control

Do not edit the hypothesis or pass/fail gate after seeing this experiment's results. If the protocol changes, create a new experiment revision or commit and record it in the manifest.

## Quantizer conformance repair before final claims

The first archived smoke/full sweep used the pre-existing arbitrary packed-code generator from bench_gpu. Those results are retained as synthetic screening, not quantizer-conformant claims. The corrected runner constructs both paths from the same absmax/7 packed cache using quantize_int4_packed_groupwise. Matrix, timing definition, historical comparator, thresholds and launch configuration are unchanged. The repair is committed before rerunning.
