# E07_long_context — Long-Context Scaling

**Status:** `READY_AFTER_E04`

## Hypothesis

Measure core packed-INT4 attention latency/throughput slope from 8K to 128K+ with Q/output transforms outside timing.

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

Report latency and effective slope; no extrapolation beyond measured context.

## Outputs

Every actual run writes to `kernel/results/E07_long_context/<run_id>/` and must contain:

- `MANIFEST.json`
- raw CSV/logs
- processed summaries only if derived from that run

## Change control

Do not edit the hypothesis or pass/fail gate after seeing this experiment's results. If the protocol changes, create a new experiment revision or commit and record it in the manifest.

## Timing boundary
This protocol uses `long-core`; the PyTorch FWHT reference is excluded. End-to-end long-context OFD timing remains blocked until E04b.
