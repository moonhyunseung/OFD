# OFD Kernel Workspace — Triton v2.3+

A private/semi-open systems research workspace for studying how rotated and compressed KV representations should be consumed during decode.

The workspace is organized around one research objective rather than a fixed venue or a predetermined stopping point:

> **Preserve the memory benefit of compressed KV representations without paying for unnecessary reconstruction or representation-mismatched execution.**

The project began from order-flipped rotated-domain decoding, but the research question has expanded as measurements exposed new execution bottlenecks. The current focus is therefore not limited to proving that order-flip is valid or that INT4 reduces KV bytes. It is to understand when a compressed representation changes the optimal decode execution strategy, which pieces of launched work are actually useful, and how the executor should be redesigned without confusing causal attribution with optimization.

## Current status

- CPU/reference contract: ready and tested
- paged/ragged/random-physical-page CPU oracle: ready
- Triton two-stage split-KV packed-INT4 decode: implemented and validated on H200
- matched paged-FP16 control path: implemented
- direct packed KV consumption without dense dequantized-cache materialization: implemented
- Qwen3.8-27B complete-model online-cache integration: completed
- E10 matched INT4/FP16 GQA-grouping intervention: completed
  - high-work packed INT4 prefers coarser q-head grouping in the tested matrix
  - matched FP16 prefers g=1 in the same matrix
  - the effect transfers to several high-work complete-model shapes
- E11 controlled interventions: completed
  - useful token work alone does not explain latency
  - launched masked work can dominate padded-capacity cases
  - 16x unique-address reduction produces only modest latency change in the tested alias control
  - no systematic adverse high-batch g1 collapse is supported by the fixed-aggregate-head repartition control
  - grouping difference-in-differences is not a direct measurement of removed HBM traffic or unpack time
- E12 inactive-split guard intervention: completed
  - skipping wholly inactive Stage-1 loops causally removes most synthetic padded-capacity overhead
  - the same guard can regress fully active paths, so it is not promoted to the default kernel
  - complete-model Qwen gains are small (~1–2% in the larger measured INT4 cases), because the online path has no wholly inactive split at the measured start points
- representation-dependent execution behavior remains supported, but no single-factor mechanism is established
- exact Stage-1 versus merge attribution and microarchitectural profiling: pending
- production serving, same-memory-budget end-to-end throughput, and optimized-native-stack superiority: not established

## Research structure

OFD should be read as three connected layers, not three independent novelty claims:

1. **Execution-order layer** — avoid reconstructing the full rotated KV cache when orthogonal algebra allows attention to consume it in its stored basis.
2. **Representation layer** — keep KV resident in a compressed packed representation and avoid dense materialization in the target path.
3. **Representation-aware execution layer** — adapt split, grouping, reuse and scheduling decisions when the compressed representation changes the cost structure of decode.

The first two layers enable the system. The third is where current measurements expose the deeper systems question: the execution geometry that is effective for FP16 need not remain optimal for packed low-bit KV, and logical useful work is not sufficient to predict GPU execution cost.

The current working view is deliberately broader than a one-variable model:

```text
latency = f(
    useful token work,
    launched/masked work,
    representation,
    execution granularity,
    physical address reuse,
    compiler/resource behavior,
    merge cost,
    hardware
)
```

This is a hypothesis space, not a fitted law. E10–E12 eliminate several simpler explanations but do not assign exact percentages to the remaining terms.

## Research growth rule

The project is not considered complete merely because the current result is publishable at some venue level. Continue the line while the next experiment answers a qualitatively new question about the same objective and materially changes the system understanding.

Do **not** add experiments only to accumulate tables or marginally improve a number. Extend the work when a new result can resolve an important uncertainty such as mechanism, boundary condition, generality, or execution law.

If a new question remains causally tied to the objective above, keep it in the OFD line even if it deepens or broadens the original formulation. If the question becomes independently motivated and no longer needs OFD's central objective, split it into a separate research line rather than forcing it into one paper.

In short:

```text
same objective + deeper understanding   -> continue
same objective + new mechanism          -> continue
same objective + broader boundary       -> continue
independent objective                    -> branch into a new project
repetitive evidence with no new insight -> stop
```

## Immediate next step

Do **not** optimize the Stage-1 Triton kernel yet.

E10–E12 show that even a small control-flow change can alter active-path performance, so editing the hot kernel before attribution risks destroying the evidence boundary through changed code generation. The existing runtime already launches Stage-1 partial generation and split merge as separate Triton kernels.

The next experiment therefore decomposes frozen paths into:

```text
Stage-1 partial generation
        +
Stage-2 split merge
        =
core decode
```

for matched FP16 and packed INT4 at representative low-, crossover- and high-work points. After phase timing, profiler runs should compare the frozen Stage-1 binaries across g1/g2/g6 and matched FP16 controls. Only then should a new kernel intervention target the attributed mechanism.

## Layout

```text
docs/          design contracts, mathematics and claim boundaries
src/           reference, Triton and runtime code
tests/         CPU/GPU validation
experiments/   frozen scientific protocols E00–E13+
results/       generated and archived measurement artifacts
tools/         runners, benchmarks, validators and analyzers
```

## Measurement discipline

The research direction may expand, but previously frozen protocols and results must not be retroactively rewritten to fit a later interpretation. New causal questions require a new experiment revision, explicit controls and preserved provenance.

Use the narrowest experiment that can distinguish competing explanations. Optimizations should follow evidence; they must not be mixed into an attribution experiment unless the optimization itself is the intervention under test.

See `experiments/EXPERIMENT_PLAN.md`, `docs/CLAIMS_AND_GATES.md`, `docs/MEASUREMENT_CONTRACT.md`, E11 findings, and `results/E10_gqa_grouping/E12_FINDINGS.md` for the current evidence boundary.
