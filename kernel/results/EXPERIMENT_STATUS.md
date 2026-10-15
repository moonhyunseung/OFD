# OFD experiment status — 2026-10-15

| Experiment | Verified status | Scope / remaining condition |
|---|---|---|
| E00 | PASS | CPU/storage/reference |
| E01 | PASS | H200 compile |
| E02 | PASS | Numerical correctness |
| E03 | Sweep completed, comparator numerical PASS | Does not establish universal high-batch collapse fix; use quantizer-conformant runs |
| E04a | PASS | Core-attention overhead |
| E04b | PASS on H200 frozen matrix | Triton GPU FWHT + attention path; not complete model/serving E2E |
| E05 | Observational sweep completed | Same-KV-budget residency and region-dependent throughput; not universal speedup |
| E06 | Sweep completed, correctness PASS | Grouped reuse speedup is region-dependent |
| E07 | Observational sweep completed | Through 128K |
| E08 | H200 revision completed | 24 launch configurations; disjoint evaluation geometric mean 1.0405x; original Blackwell plan remains archived/unmeasured |
| E09 | BLOCKED_PROTOCOL | H200 E08 prerequisite completed; real-model integration, pinned model and held-out policy evaluation remain |

Ten of eleven logical stages have completed measurements/checks under the user-authorized H200 revision; this does not mean ten performance hypotheses passed. E09 remains unfinished. See H200_E03_E05_REPORT.md and E04b_E06_REPORT.md for observed results and limitations. All new code and raw runs are kept on ofd-v2.3-measurement-safe; main is unchanged.
