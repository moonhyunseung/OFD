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
7. Which observed bottlenecks are attributable to memory traffic, unpack/dequant work, cache behavior, compiler scheduling, register pressure or occupancy?
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
next: profiler attribution and generalization, only after E10 interpretation is frozen
```

A downstream experiment must not be used for claims if its prerequisites are blocked or failed. Historical protocols remain frozen even if later work changes the interpretation of the broader research question.

## Current interpretation boundary after E10

The current evidence supports the following narrow statement:

- packed INT4 and matched FP16 do not share the same preferred GQA execution granularity under the tested Qwen geometry;
- reducing q-head repetition helps high-work packed INT4 but hurts the matched FP16 grouped path;
- this representation-dependent intervention transfers to several high-work complete-model Qwen decode shapes;
- the result is consistent with reusable packed-representation work, but current evidence does **not** isolate HBM traffic, unpack instructions, dequant arithmetic, cache residency, compiler scheduling, register pressure or occupancy as the sole mechanism;
- grouping does not close the remaining INT4-versus-FP16 execution gap.

Therefore the next attribution work should distinguish these mechanisms rather than immediately mixing in unrelated optimizations.

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

The immediate next phase should prioritize **microarchitectural attribution**, using frozen E10 paths rather than redesigning the kernel first.

Useful comparisons include representative low-, crossover- and high-work points across:

- FP16 g1 control,
- packed INT4 g1,
- packed INT4 g2,
- packed INT4 g6.

Profiler evidence should inspect at least memory traffic/cache behavior, integer/unpack activity where observable, registers/local-memory spills, occupancy, active/eligible warps and scheduler stalls.

Only after the dominant representation-dependent mechanism is identified should the next optimization be selected. That optimization must be versioned as a new experiment rather than folded back into E10.

After attribution, generalization should be chosen to test the inferred law rather than to maximize benchmark count. Candidate axes include bit-width/packing cost, GQA ratio, head dimension/model geometry, context/work regime and hardware.

## Claim discipline

- correctness experiments establish implementation equivalence, not downstream model quality;
- microbenchmarks establish only their timed scope;
- complete-model teacher-forced decode is not production serving;
- memory compression is not itself a throughput claim;
- same-memory-budget capacity is not established by host-limited or untested shapes;
- representation-dependent grouping does not by itself prove a specific microarchitectural bottleneck;
- no novelty claim should be based solely on orthogonal-rotation cancellation, rotation-based KV quantization, or generic grouped-query attention reuse without a prior-art check.

## Required run artifacts

Every claim-eligible run gets a unique directory containing `MANIFEST.json`, raw results, exact commands/environment and the source SHA. A result without reproducible provenance is not eligible for paper/company claims.

## Provenance rule

Claim-eligible experiments after correctness require a clean Git worktree unless the manifest explicitly records and justifies otherwise. Manifests record `git_dirty`, exact status, source SHA, environment and separate execution status from scientific gate status.

Later interpretation may supersede an earlier hypothesis, but raw measurements and frozen protocols must remain preserved.
