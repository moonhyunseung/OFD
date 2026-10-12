# Measurement Contract

This document defines what may and may not be inside a claim-eligible timing region.

## Core attention timing (E04a / E07)

Included:
- Triton Stage 1 packed-INT4 page traversal, direct unpack/dequant, QK, online-softmax partials
- Triton Stage 2 split-state merge

Excluded:
- Q FWHT / basis preparation
- output FWHT
- workspace allocation
- split-count calculation
- dtype/layout conversion of metadata
- debug block-table validation
- CPU/GPU synchronization other than benchmark event boundaries

A preallocated `TritonWorkspace` is mandatory for timed core execution. The hot core derives `NUM_SPLITS` from workspace shape and contains no `.item()` synchronization.

## End-to-end OFD timing (E04b)

E04b is blocked until a production GPU FWHT path exists. The current PyTorch FWHT is correctness-only and must not be used to support an end-to-end OFD latency claim.

When unblocked, E04b must include:
- Q transform when K is rotated
- packed-INT4 attention core
- output inverse transform when V is rotated

It must still exclude one-time workspace allocation and model/cache construction.

## Metadata and pointer validation

`debug_validate=True` may synchronize metadata to CPU and validate:
- positive sequence lengths
- declared `max_seq_len`
- sufficient logical block-table width
- per-request used physical-page indices within cache bounds

Debug validation is required for E02 correctness but forbidden inside claim-eligible timing loops.

Production/timed callers must provide contiguous CUDA `int32` `block_table` and `seq_lens` tensors and host-known `max_seq_len`/workspace metadata.

## Reproducibility

- frozen default seed: `1337`
- claim-eligible runs require a real Git commit and a clean worktree
- manifests record Git commit/branch/dirty status, exact commands, protocol config, environment, and artifacts
- observational experiments use `gate_status=NOT_APPLICABLE`; successful execution is not silently relabeled as a scientific PASS
