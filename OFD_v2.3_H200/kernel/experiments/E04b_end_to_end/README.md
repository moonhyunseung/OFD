# E04b_end_to_end — End-to-End OFD Cost

## Hypothesis
Measure total OFD cost only after production GPU FWHT replaces the PyTorch reference transforms.

## Timing boundary
This experiment is intentionally blocked until a production GPU FWHT path exists. The current PyTorch FWHT is correctness-only and must not be used for a claim-eligible end-to-end latency result.

## Dependencies
E04a_core_attention

## Gate
Blocked until GPU FWHT exists; then target Full-OFD/Plain end-to-end latency ratio <= 1.05.

## GPU implementation validation and measurement revision

Candidate source d1cdf1f1ebbcada2eb0d4b0d3b4c07db6506e95c passed implementation_validation/20261015T021748Z_8c57592a. GPU FWHT has 9 transform/inverse cases; grouped GQA has 24 CPU-oracle cases over D=64/128, Q/KV=1/4/8, all P/K/V/KV, randomized pages and ragged sequences. Fixed threshold 2e-3. Earlier compile failure is retained separately.

The complete frozen matrix is recorded in config.json before timing. No TMA, warp specialization, CUDA rewrite or adaptive tuning is introduced. Every performance call uses preallocated workspaces and buffers; cache construction, metadata conversions, numerical checks and CUDA-to-CPU scalar reads are outside timing.

E04b times query GPU FWHT + attention + output GPU FWHT as applicable to P/K/V/KV, with unchanged KV/P<=1.05 gate. This is full-attention-path cost, not full transformer or serving E2E. E06 compares existing q-head programs and a shared-KV grouped-query Stage 1 with identical Stage 2, data, layout and scheduling constants. GQA perf gains are observational even if the implementation gate passes.
