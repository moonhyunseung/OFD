# E13 — Frozen Stage-1 / merge phase attribution

## Question

E10–E12 show that packed INT4 and matched FP16 react differently to GQA grouping, while launched masked work, address reuse and control-flow/codegen effects also matter. Before changing the hot kernel again, determine **where** the representation-dependent latency response occurs.

The runtime already launches two kernels:

1. Stage-1 partial generation (`subgrouped_partials_kernel`)
2. Stage-2 split merge (`merge_splitkv_kernel`)

E13 measures them independently without changing either kernel.

## Primary hypothesis test

For each `(representation, g, T, B)`:

```text
L_core
L_stage1
L_merge
residual = L_core - (L_stage1 + L_merge)
```

The main question is whether the E10 grouping crossover is already present in `L_stage1`, or whether merge/workspace geometry contributes materially.

## Frozen representative shapes

| role | T | B | reason |
|---|---:|---:|---|
| low work | 2K | 1 | E10 grouping hurts INT4 |
| transition | 8K | 4 | g2 is near-neutral while coarser grouping begins to help INT4 |
| causal anchor | 32K | 4 | heavily controlled by E11/E12 |
| high work | 128K | 4 | strongest E10 grouping benefit and complete-model relevance |

Representations: `INT4`, `FP16`

Grouping: `g in {1,2,3,6}` for phase timing. Profiler follow-up may narrow to `g1/g2/g6`.

## Non-interventions

E13 must not change:

- `subgrouped_partials_kernel`
- `merge_splitkv_kernel`
- INT4 packing/quantizer
- page/group layout
- split size
- tokens per block
- warps/stages
- precision
- GQA ratio
- cache contents or physical-page mapping within a condition

No guarded E12 kernel is used for the primary E13 attribution. E13 attributes the frozen subgrouped path that established the E10 representation-dependent grouping result.

## Measurement discipline

- 3 independent worker processes
- 3 shuffled rounds per process
- 10 warmup calls per timed block
- 100 timed calls per phase block
- same workspace and exact cache buffers across scopes for a condition
- Stage-1 is run once before standalone merge timing to populate valid partials
- core output and explicit Stage-1+merge output must agree within `2e-3` relative L2 and be finite
- no outlier deletion
- raw phase measurements retained before summary

The standalone merge benchmark repeatedly merges a fixed valid partial tensor. It measures merge kernel execution cost; it is not intended to reproduce cache-state evolution.

## Interpretation

### Outcome A — Stage-1 explains the crossover

If the representation/grouping response in `L_stage1` closely matches `L_core`, proceed to microarchitectural profiling of frozen Stage-1 binaries. Merge becomes a secondary overhead.

### Outcome B — merge materially changes with grouping

If merge contributes a non-negligible fraction of the grouping effect, isolate merge/workspace geometry before attributing the phenomenon to packed-KV processing.

### Outcome C — composition fails

If `L_stage1 + L_merge` does not track `L_core` within stable measurement noise, investigate synchronization, launch composition, cache/frequency state or benchmark methodology before interpreting profiler counters.

## What E13 cannot claim

E13 does not establish HBM bandwidth, unpack/dequant instruction cost, register pressure, occupancy or scheduler stalls. It only localizes the latency response to an execution phase. Those mechanisms require the subsequent profiler experiment.
