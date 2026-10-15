# E04b/E06 H200 completion report

Source: `fce5b1b71796e79a21948b92221df123994c371e`; all measurement manifests record a clean worktree. Initial compiler failure and corrected implementation validation are retained.

## Observed results

- E04b: 32 configurations / 128 P/K/V/KV rows completed without OOM. Worst GPU-FWHT Full-OFD/Plain latency ratio 1.02566125, below the unchanged 1.05 gate. Query and output FWHT run on Triton GPU kernels inside timing when applicable. Workspace, cache construction and debug validation remain outside timing. This is complete attention-path cost, not complete transformer/serving E2E.
- E06: 36 configurations completed without OOM. Every full-case grouped-vs-q-head error <=0.002. Precondition tests additionally compare 24 ragged/random-page variant cases against CPU quantized oracle and nine GPU FWHT/inverse cases.

| Q/KV | grouped/q-head throughput gain min | max | measured cases with gain >1 |
|---|---:|---:|---:|
| 1 | 0.8404 | 1.0522 | 3/12 |
| 4 | 0.5554 | 1.6384 | 11/12 |
| 8 | 0.6331 | 1.7119 | 11/12 |

## Interpretation and limits

Grouped KV reuse is beneficial in some measured regimes and slower in others. Correctness PASS is not a universal efficiency claim. GPU FWHT removes the implementation blocker for E04b, and the frozen overhead gate passed on this H200 matrix; it does not establish model quality or complete-serving speedups. Single sweep, fixed ordering and no confidence intervals limit small differences. No Blackwell or held-out real-model run was performed.

## Remaining work

- E08: execute profiler-backed tuning on actual Blackwell. H200 is not an eligible substitute.
- E09: implement/pin the real-model serving harness, held-out calibration/evaluation and static policy, then run after E08. Synthetic traces are insufficient.

No TMA, warp specialization or CUDA rewrite was introduced. Existing E00–E07 artifacts remain. Main is unchanged.
