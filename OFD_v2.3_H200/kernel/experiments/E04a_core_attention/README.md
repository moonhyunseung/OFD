# E04a_core_attention — Core Attention Parity

## Hypothesis
Measure branch-free packed-INT4 attention cost with Q/output transforms outside the timed region.

## Timing boundary
Q preparation/FWHT and output FWHT are outside the timed region. Only Stage-1 + Stage-2 packed-INT4 attention is timed.

## Dependencies
E02_correctness

## Gate
PASS when worst Full-OFD/Plain core-attention latency ratio <= 1.05 on the frozen smoke matrix.
