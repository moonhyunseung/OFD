# Claims and gates

| Claim | Minimum gate |
|---|---|
| CPU/storage/reference correctness | E00 PASS |
| Triton implementation correctness | E02 PASS |
| Core packed-INT4 attention has <=5% OFD-vs-Plain overhead | E04a PASS |
| End-to-end OFD has <=5% overhead vs Plain INT4 | E04b PASS after production GPU FWHT exists |
| Compression improves same-budget throughput | E05 PASS with matched FP16 paged baseline |
| High-concurrency collapse is fixed | E03 PASS vs frozen v1 comparator |
| GQA implementation is efficient | E06 PASS with grouped reuse kernel |
| Long-context curve is measured | E07 COMPLETED (`gate_status=NOT_APPLICABLE`) |
| Blackwell optimization is production-relevant | E08 profiler-backed result |
| Real model quality/serving benefit | E09 held-out real-model result |

Claim-eligible experiments require a clean Git worktree. `git_dirty=true` invalidates claim eligibility even if a debug run was otherwise successful.
