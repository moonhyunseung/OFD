# OFD experiment status — 2026-10-15

| Experiment | Verified status | Scope / remaining condition |
|---|---|---|
| E00 | PASS | CPU/storage/reference |
| E01 | PASS | H200 compile |
| E02 | PASS | Numerical correctness |
| E03 | Sweep completed, comparator numerical PASS | Does not establish universal high-batch collapse fix; use quantizer-conformant runs |
| E04a | PASS | Core-attention overhead |
| E04b | PASS after wrapper correction, run 20261015T033419Z_647b462d | Triton GPU FWHT + attention path; not complete model/serving E2E |
| E05 | Observational sweep completed | Same-KV-budget residency and region-dependent throughput; not universal speedup |
| E06 | Sweep completed, correctness PASS | Grouped reuse speedup is region-dependent |
| E07 | Observational sweep completed | Through 128K |
| E08 | H200 revision completed | 24 launch configurations; disjoint evaluation geometric mean 1.0405x; original Blackwell plan remains archived/unmeasured |
| E09 | Decode layer replay smoke completed; full matrix pending data repair | H200 E08 prerequisite completed; real-model integration, pinned model and held-out policy evaluation remain |

Ten of eleven logical stages have completed measurements/checks under the user-authorized H200 revision; this does not mean ten performance hypotheses passed. E09 remains unfinished. See H200_E03_E05_REPORT.md and E04b_E06_REPORT.md for observed results and limitations. All new code and raw runs are kept on ofd-v2.3-measurement-safe; main is unchanged.


## Correction — 2026-10-15
The E04b PASS from source fce5b1b is retracted pending corrected rerun. gpu_ofd_decode unconditionally read q_work, including P/V paths where that buffer was not initialized from q. The original validator reconstructed transforms manually and missed this wrapper dispatch defect. Raw artifacts are retained for audit, not evidence of E04b parity. E06 uses the core directly and is unaffected. Regression now checks actual wrapper with poisoned scratch buffers.

Corrected E04b evidence is documented in E04B_WRAPPER_CORRECTION.md. Qwen3.8-27B real-model layer replay smoke at source 957cb0b completed 128 rows; its first full run failed before measurements due to insufficient MBPP text. v2 data construction adds official MBPP tests without recycling samples. Original failure and smoke are retained. This is not E09 serving E2E completion.
