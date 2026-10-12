# E04b_end_to_end — End-to-End OFD Cost

## Hypothesis
Measure total OFD cost only after production GPU FWHT replaces the PyTorch reference transforms.

## Timing boundary
This experiment is intentionally blocked until a production GPU FWHT path exists. The current PyTorch FWHT is correctness-only and must not be used for a claim-eligible end-to-end latency result.

## Dependencies
E04a_core_attention

## Gate
Blocked until GPU FWHT exists; then target Full-OFD/Plain end-to-end latency ratio <= 1.05.
