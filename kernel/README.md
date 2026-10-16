# OFD Kernel Workspace — Triton v2.3+

A private/semi-open systems research workspace for studying how rotated and compressed KV representations should be consumed during decode.

The workspace is organized around one research objective rather than a fixed venue or a predetermined stopping point:

> **Preserve the memory benefit of compressed KV representations without paying for unnecessary reconstruction or representation-mismatched execution.**

The project began from order-flipped rotated-domain decoding, but the research question has expanded as measurements exposed new execution bottlenecks. The current focus is therefore not limited to proving that order-flip is valid or that INT4 reduces KV bytes. It is to understand when a compressed representation changes the optimal decode execution strategy, and to redesign the executor accordingly.

## Current status

- CPU/reference contract: ready and tested
- paged/ragged/random-physical-page CPU oracle: ready
- Triton two-stage split-KV packed-INT4 decode: implemented and validated on H200
- matched paged-FP16 control path: implemented
- direct packed KV consumption without dense dequantized-cache materialization: implemented
- split-KV scaling experiments: completed
- grouped/subgrouped GQA reuse: implemented and causally evaluated through E10
- Qwen3.8-27B complete-model online-cache integration: completed
- E10 low-level causal sweep: completed with matched INT4/FP16 grouping controls
- E10 complete-model confirmation: completed with a policy frozen before model evaluation
- representation-dependent execution behavior: observed; high-work packed INT4 benefits from reduced q-head repetition while matched FP16 does not
- exact microarchitectural attribution: pending profiler work
- production serving, same-memory-budget end-to-end throughput, and optimized-native-stack superiority: not established

## Research structure

OFD should be read as three connected layers, not three independent novelty claims:

1. **Execution-order layer** — avoid reconstructing the full rotated KV cache when orthogonal algebra allows attention to consume it in its stored basis.
2. **Representation layer** — keep KV resident in a compressed packed representation and avoid dense materialization in the target path.
3. **Representation-aware execution layer** — adapt split, grouping, reuse and scheduling decisions when the compressed representation changes the cost structure of decode.

The first two layers enable the system. The third is where current measurements are exposing the deeper systems question: the execution geometry that is effective for FP16 need not remain optimal for packed low-bit KV.

## Research growth rule

The project is not considered complete merely because the current result is publishable at some venue level. Continue the line while the next experiment answers a qualitatively new question about the same objective and materially changes the system understanding.

Do **not** add experiments only to accumulate tables or marginally improve a number. Extend the work when a new result can resolve an important uncertainty such as mechanism, boundary condition, generality, or execution law.

If a new question remains causally tied to the objective above, keep it in the OFD line even if it deepens or broadens the original formulation. If the question becomes independently motivated and no longer needs OFD's central objective, split it into a separate research line rather than forcing it into one paper.

In short:

```text
same objective + deeper understanding  -> continue
same objective + new mechanism         -> continue
same objective + broader boundary      -> continue
independent objective                   -> branch into a new project
repetitive evidence with no new insight -> stop
```

## Layout

```text
docs/          design contracts, mathematics and claim boundaries
src/           reference, Triton and runtime code
tests/         CPU/GPU validation
experiments/   frozen scientific protocols E00–E10+
results/       generated and archived measurement artifacts
tools/         runners, benchmarks, validators and analyzers
```

## Measurement discipline

The research direction may expand, but previously frozen protocols and results must not be retroactively rewritten to fit a later interpretation. New causal questions require a new experiment revision, explicit controls and preserved provenance.

Use the narrowest experiment that can distinguish competing explanations. Optimizations should follow evidence; they must not be mixed into an attribution experiment unless the optimization itself is the intervention under test.

See `experiments/EXPERIMENT_PLAN.md`, `docs/CLAIMS_AND_GATES.md`, `docs/MEASUREMENT_CONTRACT.md`, and the archived E10 findings for the current evidence boundary.
