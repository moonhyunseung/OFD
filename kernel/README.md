# OFD Kernel Workspace — Triton v2.3+

A private/semi-open systems research workspace for studying how rotated and compressed KV representations should be represented and consumed during decode.

The project is organized around one research objective:

> **Preserve the memory benefit of compressed KV representations without paying for unnecessary reconstruction or representation-mismatched execution.**

The current phase goes one step deeper than executor tuning. E10–E12 showed that the correctness-first packed-INT4 kernel has representation-dependent execution behavior, but those measurements do not prove that its current dequant-first contraction order is fundamental. The next question is whether the attention operator itself can be factored into a cheaper exact or controlled-approximate form that maps directly to low-bit GPU primitives.

## Current status

- CPU/reference contract: ready and tested
- paged/ragged/random-physical-page CPU oracle: ready
- Triton two-stage split-KV packed-INT4 decode: implemented and validated on H200
- matched paged-FP16 control path: implemented
- direct packed KV cache consumption without dense cache materialization: implemented
- Qwen3.8-27B complete-model online-cache integration: completed
- E10 matched INT4/FP16 GQA-grouping intervention: completed
  - high-work packed INT4 prefers coarser q-head grouping in the tested matrix
  - matched FP16 prefers g=1 in the same matrix
- E11 controlled interventions: completed
  - useful token work alone does not explain latency
  - launched masked work can dominate padded-capacity cases
  - simple footprint-proportional and high-batch-collapse explanations are not supported by the tested controls
- E12 inactive-split guard intervention: completed
  - skipping wholly inactive Stage-1 loops removes most synthetic padded-capacity overhead
  - the same guard can regress fully active paths
  - complete-model gains are small because the measured online path has no wholly inactive split at its starting points
- E13 phase-attribution tooling: implemented, retained as an optional diagnostic for the old executor
- **operator/circuit re-derivation: active research direction**
- E14 operator-structure protocol: defined, no results yet
- E15 circuit-mapped contraction prototype: not started
- production serving, same-memory-budget end-to-end throughput, and superiority to optimized native stacks: not established

## Current research structure

OFD is best read as four connected layers:

1. **Execution-order layer** — use algebraic invariance to avoid cache-wide reconstruction/inverse transforms.
2. **Representation layer** — keep KV in a compressed packed representation.
3. **Operator-geometry layer** — identify exact quotient/null spaces, basis symmetries, decompositions, movable operators and controlled approximation regimes of attention itself.
4. **Circuit-mapping layer** — choose an equivalent factorization whose natural dimensions/types map efficiently to real GPU low-bit matrix primitives.

The present work is concentrated on layers 3–4.

## Key mathematical shift

For one KV head and its GQA query group, define feature space `F=R^d`, token space `T=R^L`, query-head space `H=R^r`.

```text
Q in Hom(F,H)
K,V in Hom(F,T)
score: Q K^T
value: P V
```

For fixed Q, keys are observed only modulo `ker(Q)`. For the current `r=6,d=256` geometry, one-step query rank is at most 6 and instantaneous feature nullity is at least 250. Softmax adds an exact row-constant quotient in token space. For fixed P, V is observed only modulo `ker(P)` in the token axis.

These are operator facts, not claims that 250 persistent KV dimensions can be deleted. E14 asks whether any of this instantaneous structure is stable enough to become a real computational reduction.

## Exact packed-INT4 algebra

The current storage encodes symmetric values `q in [-7,7]` as unsigned nibbles `U=q+8` with one scale per 32-wide group. Instead of reconstructing a dense K group first,

```text
Q_g K_g^T
= (Q_g U_g^T) D(s_g)
  - 8 (Q_g 1) s_g^T.
```

Thus group scale can move after the width-32 contraction and the current unsigned-code correction is rank 1. An analogous identity holds for V.

This makes the current GQA-grouping crossover potentially an artifact of the existing dequant-first contraction order rather than a final execution law.

## Hadamard / basis interpretation

Normalized Hadamard is an exact orthogonal feature-space automorphism:

```text
(QH)(KH)^T = QK^T.
```

It preserves rank/nullity and does **not** reduce dimension by itself. Its role is to change coordinate geometry/coherence before nonlinear quantization.

For `256=8*32`, the Hadamard structure can be separated into cross-block and within-32-block mixing. Basis placement relative to RoPE is a conjugation problem: moving a transform across RoPE is only attractive when the conjugated RoPE remains cheap/structured.

## Hardware interpretation

Hopper/PTX exposes signed/unsigned 4-bit MMA families with reduction dimension 32, and CUTLASS documents Hopper mixed-input kernels involving int4 and f16/bf16. Therefore the current 32-wide quantization group is a plausible circuit-aligned reduction width.

This is **not** yet a speed claim. `r=6` still has to map efficiently to M/N tile shapes; conversion, scales, register use and generated PTX/SASS must be accounted for explicitly.

## Immediate next step

Do **not** start with another broad timing/profiler sweep of the old kernel.

Run E14 operator-structure probes first:

```text
E14A  query rank/effective rank + subspace drift + 32-wide block capture
E14B  basis/coherence + scale spectrum + Q integer/radix plane requirement
E14C  V probability/effective-support + output-visible error
E14D  static byte/instruction/MMA/tile work ledger
```

Only if E14 identifies a concrete structural reduction should E15 create a new hot-kernel lineage. E15 must freeze the algebra/work ledger first, then pass numerical gates, inspect generated PTX/SASS, and only then use latency as confirmation.

## Research growth rule

Continue while a new step resolves an exact equivalence, exploitable null/quotient structure, controlled approximation regime, circuit mapping, boundary condition or generalizable execution principle.

Do not add experiments merely to produce another timing table for the same executor.

```text
same objective + deeper operator understanding -> continue
same objective + circuit-mapped reduction       -> continue
same objective + broader verified boundary      -> continue
independent objective                            -> branch
repetitive evidence with no structural insight  -> stop
```

## Layout

```text
docs/          mathematical/operator contracts, architecture and claim boundaries
src/           reference, Triton and runtime code
tests/         CPU/GPU validation
experiments/   frozen scientific protocols E00+
results/       generated/archived evidence
tools/         runners, benchmarks, validators and analyzers
```

Primary current documents:

- `docs/OPERATOR_GEOMETRY_AND_CIRCUIT.md`
- `docs/MATH.md`
- `docs/ARCHITECTURE.md`
- `experiments/EXPERIMENT_PLAN.md`
- `experiments/E14_operator_structure/README.md`

## Measurement discipline

Historical E00–E13 protocols/results remain frozen. A new storage encoding, basis policy, quantizer, contraction order or hardware-native kernel belongs to a new lineage.

Exact algebra must be distinguished from conditional equivalence and approximation. Approximation must state the operator-visible error norm/bound. Hardware claims require generated-code evidence. Timing is the final confirmation, not the source of the mathematical mechanism.
