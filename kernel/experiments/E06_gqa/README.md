# E06_gqa — GQA KV-Reuse

**Status:** `BLOCKED_GQA_REUSE`

## Hypothesis

Compare current q-head programs against grouped Q-head KV reuse for Q/KV = 1,4,8.

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

Requires grouped-GQA Triton kernel implementation.

## Outputs

Every actual run writes to `kernel/results/E06_gqa/<run_id>/` and must contain:

- `MANIFEST.json`
- raw CSV/logs
- processed summaries only if derived from that run

## Change control

Do not edit the hypothesis or pass/fail gate after seeing this experiment's results. If the protocol changes, create a new experiment revision or commit and record it in the manifest.

## GPU implementation validation and measurement revision

Candidate source d1cdf1f1ebbcada2eb0d4b0d3b4c07db6506e95c passed implementation_validation/20261015T021748Z_8c57592a. GPU FWHT has 9 transform/inverse cases; grouped GQA has 24 CPU-oracle cases over D=64/128, Q/KV=1/4/8, all P/K/V/KV, randomized pages and ragged sequences. Fixed threshold 2e-3. Earlier compile failure is retained separately.

The complete frozen matrix is recorded in config.json before timing. No TMA, warp specialization, CUDA rewrite or adaptive tuning is introduced. Every performance call uses preallocated workspaces and buffers; cache construction, metadata conversions, numerical checks and CUDA-to-CPU scalar reads are outside timing.

E04b times query GPU FWHT + attention + output GPU FWHT as applicable to P/K/V/KV, with unchanged KV/P<=1.05 gate. This is full-attention-path cost, not full transformer or serving E2E. E06 compares existing q-head programs and a shared-KV grouped-query Stage 1 with identical Stage 2, data, layout and scheduling constants. GQA perf gains are observational even if the implementation gate passes.
