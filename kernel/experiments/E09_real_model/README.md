# E09_real_model — Real-Model Serving

**Status:** `BLOCKED_REAL_MODEL`

## Hypothesis

Integrate Qwen/Llama real KV, held-out layer policy, and serving metrics.

## Dependencies

E05_same_memory_budget, E06_gqa, E08_blackwell_tuning

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

Requires real-model integration and held-out policy validation.

## Outputs

Every actual run writes to `kernel/results/E09_real_model/<run_id>/` and must contain:

- `MANIFEST.json`
- raw CSV/logs
- processed summaries only if derived from that run

## Change control

Do not edit the hypothesis or pass/fail gate after seeing this experiment's results. If the protocol changes, create a new experiment revision or commit and record it in the manifest.
