# OFD Triton v2+ — Experiment Plan

## Research objective

OFD studies a systems question:

> **How should a decode engine consume rotated and compressed KV representations so that their memory benefit is not lost to reconstruction, materialization, or representation-mismatched execution?**

The project began by testing whether rotated-domain attention could avoid repeated cache-wide inverse transforms. That algebraic step is an enabling mechanism, not the final research objective. The current line extends naturally to direct packed-KV consumption, split execution, GQA reuse and representation-aware scheduling whenever those questions are needed to understand where compression benefits are preserved or lost.

The research objective is therefore broader than reaching one fixed kernel speed target. A result should be extended when the next experiment can change our understanding of the mechanism, boundary, generality, or optimal execution policy. A venue threshold is not a scientific stopping criterion.

## Primary questions

1. Can paged packed-KV attention remain numerically faithful to the frozen reference without dense dequantized KV materialization?
2. Which costs are removed by order-flipped execution, and which costs remain after cache-wide reconstruction is eliminated?
3. How do context length, batch, split geometry and GQA alter the execution cost of packed low-bit KV?
4. Does a compressed representation change the optimal execution granularity relative to a matched FP16 path?
5. Which representation-specific costs are reusable across q-heads or other execution units, and when does reuse outweigh lost parallelism or increased resource pressure?
6. Can any low-level execution gain survive complete-model decode rather than only a microbenchmark?
7. Which observed bottlenecks are attributable to useful traversal, launched masked work, merge cost, memory traffic, unpack/dequant work, cache behavior, compiler scheduling, register pressure or occupancy?
8. How general are the observed execution regimes across representation precision, GQA ratio, model geometry and hardware?
9. Under an equal KV-memory budget, can the final representation-aware executor convert compression into a real throughput/capacity advantage without hiding costs?

## Experiment lineage

```text
E00 CPU reference
  ↓
E01 GPU compile smoke
  ↓
E02 numerical correctness
  ├─→ E03 split scaling
  ├─→ E04a core attention parity
  │       └─→ E04b GPU-transform / end-to-end OFD overhead
  ├─→ E05 same-memory-budget infrastructure
  └─→ E06 grouped-GQA reuse
          ↓
E07 long-context core scaling
          ↓
E08 hardware/kernel tuning
          ↓
E09 real Qwen3.8-27B complete-model decode
          ↓
E10 matched INT4/FP16 GQA-grouping causal intervention
          ↓
E11 controlled causal interventions
  - capacity / launched masked work
  - unique physical-address footprint
  - batch/head repartition
  - Q/KV repetition
          ↓
E12 inactive-split loop guard intervention
  - synthetic padded-capacity causal test
  - active-path regression control
  - complete-model confirmation
          ↓
E13 phase attribution
  - frozen Stage-1 timing
  - frozen split-merge timing
  - core = Stage-1 + merge accounting
          ↓
profiler attribution on frozen Stage-1 binaries
          ↓
mechanism-targeted kernel intervention
          ↓
generalization of the inferred execution law
```

A downstream experiment must not be used for claims if its prerequisites are blocked or failed. Historical protocols remain frozen even if later work changes the interpretation of the broader research question.

## Current interpretation boundary after E12

The evidence now supports a stronger but still bounded picture:

- packed INT4 and matched FP16 do not share the same preferred GQA execution granularity under the tested Qwen geometry;
- reducing q-head repetition helps high-work packed INT4 but hurts the matched FP16 grouped path in E10;
- no systematic independent adverse high-batch g1 penalty is supported by the tested equal-work/repartition controls;
- useful token traversal alone is insufficient to explain latency: E11 changes declared capacity while holding active Q/KV and sequence length fixed and observes large latency changes from additional launched masked work;
- reducing unique referenced physical KV addresses by 16x produces only modest latency movement in the tested alias control, rejecting a simple footprint-proportional latency model within that matrix;
- E12 causally confirms that wholly inactive Stage-1 loops can dominate synthetic padded-capacity cases, but the same guard can regress fully active paths;
- the complete-model path has no wholly inactive split at the measured starting points, so the large synthetic E12 speedup does not explain the E09/E10 high-work complete-model gap;
- grouping difference-in-differences is not a direct measurement of removed HBM traffic, unpack time or dequant arithmetic;
- representation-dependent execution remains a live explanation, but compiler/resource/scheduling interactions remain unresolved.

Therefore the next step must separate **where** the latency difference occurs before changing **how** the kernel executes.

## E13 phase-attribution rule

E13 is an attribution experiment, not an optimization experiment.

The current runtime already launches Stage-1 partial generation and Stage-2 merge as separate Triton kernels. E13 must therefore preserve the existing frozen kernel sources and measure the phases independently:

```text
L_core(rep, g, shape)
L_stage1(rep, g, shape)
L_merge(rep, g, shape)
residual = L_core - (L_stage1 + L_merge)
```

The residual is a measurement sanity diagnostic for launch/timing composition; it is not automatically interpretable as a third kernel cost.

Representative conditions should include at least:

- a low-work condition where grouping loses for INT4;
- a transition/crossover condition;
- a high-work condition where grouping strongly helps INT4;
- matched FP16 controls at the same shapes and grouping values.

Primary comparisons:

- FP16 g1;
- FP16 grouped controls matching the INT4 g values;
- packed INT4 g1;
- packed INT4 g2;
- packed INT4 g6 where valid.

E13 must use the same cache buffers, workspace geometry, split size, token tile, warps/stages and representation layouts as the frozen source being attributed. No nibble optimization, split-size tuning, warp tuning, new guard, layout change, precision change or quantizer change is allowed.

### E13 decision outcomes

If grouping gain is almost entirely in Stage-1:
- profile Stage-1 directly and treat merge as a secondary cost.

If merge changes materially with grouping/capacity:
- isolate merge as its own systems problem before attributing Stage-1 behavior.

If `L_stage1 + L_merge` does not track core timing within expected launch/timing noise:
- inspect benchmark synchronization/composition before interpreting either phase.

Only after this decomposition is frozen should profiler evidence be used to distinguish memory/cache, integer/unpack activity, register pressure/spills, occupancy and scheduler stalls.

## Frozen common contract for the existing lineage

Unless a new experiment explicitly declares a new lineage or intervention:

- group size = 32
- page size = 16 unless explicitly varied
- signed INT4 storage, encoded as `q+8`, low nibble first
- symmetric `absmax/7` scale; normal finite quantization uses `[-7,7]`
- scale dtype = FP16
- FP32 online-softmax / split merge accumulators
- randomized logical→physical page mapping for correctness
- no dense dequantized KV materialization in the Triton target path
- previously frozen split/tile/warp/stage settings remain unchanged during causal attribution unless the setting itself is the intervention

## Research growth / stopping rule

Do not stop the research merely because the present result is sufficient for a particular publication tier. Stop or branch based on the scientific structure.

Continue when:

- the next question follows from the same central objective;
- answering it can qualitatively change the interpretation;
- it resolves a mechanism, boundary condition or generalization gap;
- it can turn an implementation-specific observation into a reproducible execution principle.

Branch into a separate project when:

- the new question has an independent objective;
- OFD is no longer necessary to motivate the problem;
- the experiments, baselines and claims would remain meaningful without the OFD representation/execution question.

Stop extending the current line when additional work only repeats an established trend, adds marginal table coverage without reducing uncertainty, or optimizes a number without teaching us anything new about the objective.

## Next-phase decision rule

Do not redesign the hot Stage-1 kernel before E13 and profiler attribution are frozen. E12 already demonstrates that a logically beneficial control-flow change can alter active-path performance, so any premature kernel edit can change code generation and destroy the causal comparison.

After E13, profiler evidence should inspect at least:

- DRAM/L2 traffic and cache behavior;
- integer/bitwise/unpack/conversion activity where observable;
- registers per thread and local-memory spills;
- achieved occupancy;
- active and eligible warps;
- scheduler/warp-stall reasons;
- CTA/wave behavior across g1/g2/g6;
- whether representation-dependent differences move with Stage-1 only or also with merge/workspace geometry.

Only after the dominant mechanism is narrowed should the next kernel intervention be selected. That intervention must be versioned as a new experiment rather than folded back into E10–E13.

After attribution, generalization should test the inferred law rather than maximize benchmark count. Candidate axes include bit-width/packing cost, GQA ratio, head dimension/model geometry, context/work regime and hardware.

## Claim discipline

- correctness experiments establish implementation equivalence, not downstream model quality;
- microbenchmarks establish only their timed scope;
- complete-model teacher-forced decode is not production serving;
- memory compression is not itself a throughput claim;
- same-memory-budget capacity is not established by host-limited or untested shapes;
- representation-dependent grouping does not by itself prove a specific microarchitectural bottleneck;
- inactive-split skipping does not imply a production serving gain when the production path has no inactive split;
- no novelty claim should be based solely on orthogonal-rotation cancellation, rotation-based KV quantization, or generic grouped-query attention reuse without a prior-art check.

## Required run artifacts

Every claim-eligible run gets a unique directory containing `MANIFEST.json`, raw results, exact commands/environment and the source SHA. A result without reproducible provenance is not eligible for paper/company claims.

## Provenance rule

Claim-eligible experiments after correctness require a clean Git worktree unless the manifest explicitly records and justifies otherwise. Manifests record `git_dirty`, exact status, source SHA, environment and separate execution status from scientific gate status.

Later interpretation may supersede an earlier hypothesis, but raw measurements and frozen protocols must remain preserved.
