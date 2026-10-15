# E08 H200 launch tuning

Source: `06b1dd85d640af74e8917ad4131e5bed234f8892`, clean worktree. Full run: `E08_h200_tuning/20261015T023501Z_c0415b41`; separate smoke retained. H200, CUDA 13.0, PyTorch 2.14.1+cu130, Triton 3.8.0; full environment in manifests. CPU regression: 16 tests passed.

## Direct observations

All 24 existing launch configurations completed calibration (3 cases, 3 shuffled timing rounds), followed by evaluation on 4 disjoint cases using only the globally selected configuration and frozen baseline. 240 raw measurement rows, no OOM, maximum candidate-versus-baseline relative error 1.27913e-5 <= 2e-3.

Baseline: split1024/tile16/warps4/stages2. Selected: split512/tile16/warps8/stages1. Calibration geometric mean speedup 1.37318x. Selection was written to selection.json before evaluation began. No production default changed.

| Evaluation context | Batch | Q/KV | Baseline median ms | Selected median ms | Throughput ratio |
|---|---:|---:|---:|---:|---:|
| 8K | 8 | 4 | 1.19562 | 1.16934 | 1.02247 |
| 32K | 1 | 1 | 0.17667 | 0.15574 | 1.13439 |
| 64K | 4 | 1 | 1.25514 | 1.27893 | 0.98140 |
| 128K | 8 | 4 | 18.61683 | 18.07701 | 1.02986 |

Held-out evaluation geometric mean: **1.04054x**, approximately 4.05% throughput improvement across these four equally weighted cases. One case regressed by approximately 1.86%; this is not a universal speedup. Smoke numbers are excluded.

## Interpretation and limits

The chosen global launch parameters yield modest generalization compared with their calibration benefit. Configuration tuning alone has not removed the larger high-context/GQA costs established by prior sweeps. These are measurements of the existing query-head core and its event timing contract; Python launch overhead/starvation may contribute. They do not measure grouped-GQA tuning, FP16 matched-budget retuning, GPU-FWHT E2E retuning, or production serving. No confidence interval or independent process replication is available. Hardware profiler counters were not collected, so memory bandwidth, occupancy and dequant instruction bottlenecks are hypotheses rather than directly established causes.

## Next candidates

Profile the 128K/GQA case before architecture changes, measure grouped-GQA with equivalent tuning, and replicate these fixed configurations in independent processes. Preserve baseline defaults until broader workload validation. E09 still needs real-model integration, a pinned model and separate calibration/evaluation data; its Blackwell dependency was replaced with this H200 stage without weakening those requirements. The historical Blackwell protocol/results remain unchanged and are not claimed complete.
