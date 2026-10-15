# Blackwell notes

Primary optimization target: RTX PRO 6000 Blackwell. The design remains architecture-portable; Blackwell-specific fast paths are introduced only after baseline profiling.

## Candidate knobs

- split size
- tokens per block
- num warps / stages
- q-head tile under GQA
- partial workspace size/layout
- vectorized INT4 unpack

## Deferred fast paths

- TMA-assisted movement
- warp specialization
- persistent scheduling / multi-CTA strategies
- fused/in-register FWHT

Do not add these before E02 correctness and baseline profiler traces are frozen. Optimization must respond to an observed bottleneck, not a guessed one.
