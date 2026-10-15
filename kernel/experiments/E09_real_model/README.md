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

## H200 dependency revision (2026-10-15)

The user authorized H200 as the remaining execution target. The logical E08 prerequisite is now `E08_h200_tuning`; the archived Blackwell plan is not a prerequisite and remains unmeasured. This revision does not waive real-model serving integration or held-out policy validation. E08 core tuning alone is insufficient to complete E09.

## Qwen3.8-27B decode-only revision

Pinned model/data and sample construction are in `decode_protocol.json`. Four domains: WikiText prose, MBPP code, GSM8K mathematics and LongBench Qasper documents. Decode only: prefill creates state and is excluded from timing. Phase A measures actual post-RoPE queries/KV from all 16 full-attention layers (D256, Q/KV6). The other 48 Gated DeltaNet layers remain native. Its native model forward produces real decode states; per-layer P/K/V/KV replay uses matched paged FP16 baseline and preallocated GPU-FWHT attention timing. This is not whole-model E2E and does not measure KV append or capacity. Phase B requires online packed-cache integration before E09 can be declared complete. `run_e09_layers.py` runs a separate smoke, then the full fixed matrix only if smoke passes. Raw files are flushed per layer; each run has an exact clean source SHA, environment, command, data IDs/checksums, policy, CSV and log. Do not confuse INT4-versus-FP16 quality degradation with implementation correctness tolerance.
