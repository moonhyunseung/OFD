# E05_same_memory_budget — Same-Memory-Budget Throughput

**Status:** `BLOCKED_FP16_BASELINE`

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
