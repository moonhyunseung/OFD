# E05_same_memory_budget — Same-Memory-Budget Throughput

**Status:** `READY_MATCHED_PAGED_CORE`

## Hypothesis

Compare aggregate throughput under equal KV-memory budget.

## Dependencies

E04_plain_vs_ofd

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

Requires production-like FP16 paged baseline in the same harness; freeze batch-matching rule before run.

## Outputs

Every actual run writes to `kernel/results/E05_same_memory_budget/<run_id>/` and must contain:

- `MANIFEST.json`
- raw CSV/logs
- processed summaries only if derived from that run

## Change control

Do not edit the hypothesis or pass/fail gate after seeing this experiment's results. If the protocol changes, create a new experiment revision or commit and record it in the manifest.

## Pre-measurement protocol revision: E05_core_paged_v1

The original E05 dependency on blocked E04b is retained in Git history. This revision answers the user's explicitly requested core-only memory-budget comparison, with dependencies on E03 and E04a. It does not unblock or replace E04b. No GPU FWHT or real-model end-to-end claim is made.

FP16 and INT4 use the same paged online-softmax loop, page size, head geometry, split size 1024, tile 16, four warps/two stages, and identical second-stage merge. FP16 directly loads [physical page, KV head, token, D] values; INT4 loads packed values plus FP16 scales. Both use FP16 query/output and FP32 accumulation.

Frozen pairing uses integer B_INT4=(32*B_FP16)//9. Actual KV buffer bytes, workspace bytes, latencies, token rates, resident batch ratio, environment and Git SHA are recorded. This is KV-budget matching, not total device-memory matching. Buffer residency is directly allocated, but maximum serving concurrency is not measured.

The common first requests share original FP16 random values and block-table entries. INT4 uses the existing absmax/7 quantizer during untimed setup. Additional INT4 requests use independent seeded synthetic data. Correctness is tested against a dense FP32 oracle only outside timing with ragged randomized pages. Sweep finite checks are also outside timing.

Matrix: contexts 8K/16K/32K/64K/128K; FP16 B=1/2/4/8; INT4 B=3/7/14/28; Hkv=8, Q/KV=1, D=128, page=16, group=32. Performance outcomes remain observational even when INT4 loses. This baseline is structurally matched; it is not a tuned production serving engine.
