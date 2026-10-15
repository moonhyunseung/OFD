# v2.3 Review Fixes

This checkpoint applies the pre-Blackwell measurement/reproducibility review.

1. Removed timed-path split-count CUDA->CPU synchronization when workspace metadata is supplied.
2. Removed `seq_lens.any()` synchronization from the hot validation path.
3. Split OFD timing into E04a core-attention parity and E04b end-to-end cost; E04b is blocked until production GPU FWHT exists.
4. Added debug-only ragged/block-table/physical-page validation without penalizing performance runs.
5. Added frozen benchmark seeds and records them in output CSVs.
6. Added clean-Git provenance gates and manifest fields (`git_available`, `git_dirty`, porcelain status, claim eligibility).
7. Separated `execution_status` from `gate_status`; observational E07 uses `NOT_APPLICABLE` rather than a false PASS.
8. Unified `MANIFEST_TEMPLATE.json` with runtime manifest schema v2.
9. Changed E07 to core-only long-context measurement while end-to-end FWHT remains blocked.
10. Moved installer backups outside the worktree and refuse overwrite of a dirty existing worktree.
11. Added protocol regression tests; CPU suite now contains 16 tests.
