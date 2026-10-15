# Claims and gates

| Claim | Minimum gate |
|---|---|
| CPU/storage/reference correctness | E00 PASS |
| Triton implementation correctness | E02 PASS |
| Core packed-INT4 attention has <=5% OFD-vs-Plain overhead | E04a PASS |
| GPU query-FWHT + attention + output-FWHT has <=5% overhead vs Plain INT4 on the frozen matrix | E04b measured worst KV/P <=1.05 after GPU FWHT correctness; this is not full-model serving E2E |
| Compression improves same-KV-budget throughput in a stated region | E05 per-case measured throughput ratio >1 with matched paged FP16 baseline; residency alone is insufficient |
| High-concurrency collapse is fixed | Reproduce collapse in the frozen v1 control and demonstrate improved batch scaling; E03 numerical PASS alone is insufficient |
| Grouped GQA is faster in a stated region | E06 correctness PASS plus measured grouped/q-head speedup >1 in that region; correctness alone is insufficient |
| Long-context curve is measured | E07 COMPLETED (`gate_status=NOT_APPLICABLE`) |
| Blackwell optimization is production-relevant | E08 profiler-backed result |
| Real model quality/serving benefit | E09 held-out real-model result |

Claim-eligible experiments require a clean Git worktree. `git_dirty=true` invalidates claim eligibility even if a debug run was otherwise successful.
