# E03_split_scaling — Split-KV Scaling

**Status:** `BLOCKED_COMPARATOR`

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

Requires a frozen serial-v1 comparator in the same harness before execution.

## Outputs

Every actual run writes to `kernel/results/E03_split_scaling/<run_id>/` and must contain:

- `MANIFEST.json`
- raw CSV/logs
- processed summaries only if derived from that run

## Change control

Do not edit the hypothesis or pass/fail gate after seeing this experiment's results. If the protocol changes, create a new experiment revision or commit and record it in the manifest.
