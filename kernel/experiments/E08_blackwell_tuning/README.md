# E08_blackwell_tuning — Blackwell Tuning

**Status:** `BLOCKED_GATES`

## Hypothesis

Tune split/tile/warps/stages, then consider TMA/warp specialization only after architecture gates pass.

## Dependencies

E03_split_scaling, E04_plain_vs_ofd, E07_long_context

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

No TMA/warp-specialization claim before baseline profiler evidence.

## Outputs

Every actual run writes to `kernel/results/E08_blackwell_tuning/<run_id>/` and must contain:

- `MANIFEST.json`
- raw CSV/logs
- processed summaries only if derived from that run

## Change control

Do not edit the hypothesis or pass/fail gate after seeing this experiment's results. If the protocol changes, create a new experiment revision or commit and record it in the manifest.
