# OFD Triton v2+ — Experiment Plan

## Research objective

OFD studies a systems/mathematical design question:

> **How should a decode engine represent and evaluate rotated/compressed KV attention so that exact operator symmetries, null spaces, factorization structure, and hardware-native low-bit contractions are used before paying reconstruction or representation-mismatched execution costs?**

The original objective remains intact:

> **Preserve the memory benefit of compressed KV representations without paying for unnecessary reconstruction or representation-mismatched execution.**

What changed after E10–E12 is the methodology. The next phase should not treat the current packed-INT4 Triton loop as the fixed object and search empirically for a faster schedule. The current executor is one implementation of a richer operator. Before another hot-kernel optimization, derive which dimensions/operations are actually necessary, which are exact equivalence classes, which approximations have operator-level bounds, and which resulting contractions match the target GPU circuit.

See `docs/OPERATOR_GEOMETRY_AND_CIRCUIT.md` for the full working derivation.

---

## Primary questions

### Existing-line questions already addressed in part by E00-E12

1. Can paged packed-KV attention remain numerically faithful without dense dequantized-cache materialization?
2. Which costs are removed by order-flipped execution and which remain?
3. How do context, split geometry, GQA grouping, launched masked work and representation affect the current executor?
4. Can low-level execution gains survive complete-model decode?

### New operator/circuit questions

5. What are the exact domain/codomain, kernel, image and quotient spaces of K/V attention at one decode step?
6. Which basis changes are exact symmetries of the score/value operators, and where can they be moved through RoPE/neighboring linear maps by commutation, conjugation, adjoints or absorption?
7. Which parts of current groupwise INT4 dequantization can be moved outside the inner contraction or represented as low-rank corrections exactly?
8. Is the large instantaneous query-induced feature nullity stable enough across decode steps/layers to become a persistent computational reduction?
9. Can query-visible energy be concentrated into a small number of 32-wide blocks that align with low-bit hardware contraction dimensions?
10. Does Hadamard/other structured mixing reduce **circuit work** (conversion/plane count/tile count), not merely raw quantization MSE?
11. How concentrated is V work in token space after softmax, and can any support reduction be given an explicit output-error bound?
12. After the mathematical reductions are fixed, does the predicted byte/instruction/MMA reduction appear in generated PTX/SASS and measured latency?

---

## Experiment lineage

Historical protocols remain frozen.

```text
E00 CPU reference
  ↓
E01 GPU compile smoke
  ↓
E02 numerical correctness
  ├─→ E03 split scaling
  ├─→ E04a core attention parity
  │       └─→ E04b transform/end-to-end overhead
  ├─→ E05 same-memory-budget infrastructure
  └─→ E06 grouped-GQA reuse
          ↓
E07 long-context core scaling
          ↓
E08 hardware/kernel tuning
          ↓
E09 Qwen3.8-27B complete-model decode
          ↓
E10 matched INT4/FP16 GQA-grouping intervention
          ↓
E11 controlled interventions
          ↓
E12 inactive-split guard + complete-model control
          ├──────────────→ E13 phase timing (diagnostic, optional)
          │
          └──────────────→ operator/circuit re-derivation
                              ↓
                         E14 operator structure
                         ├─ E14A query visible subspace
                         ├─ E14B basis/quantizer geometry
                         ├─ E14C V token/output geometry
                         └─ E14D static circuit accounting
                              ↓
                         E15 circuit-mapped contraction prototype
                              ↓
                         complete-model / same-memory-budget validation
                              ↓
                         generalization only after the law/design is fixed
```

E13 is retained because Stage-1/merge timing may still be useful for diagnosis of the old executor. It is **not** a scientific prerequisite for E14 or the new kernel design. No E13 result should be elevated into the new mechanism without operator/circuit justification.

---

## Current evidence boundary after E12

The following remain supported for the **current executor**:

- packed INT4 and matched FP16 prefer different GQA execution granularities in the tested E10 matrix;
- high-work packed INT4 benefits from coarser grouping while the matched FP16 grouped path does not;
- no systematic independent adverse high-batch g1 penalty is supported by the equal-work/repartition controls;
- useful token work alone does not explain latency;
- launched masked work can dominate synthetic padded-capacity cases;
- a 16x reduction in unique referenced addresses causes only modest movement in the E11 alias control, rejecting a simple footprint-proportional latency model in that matrix;
- skipping wholly inactive Stage-1 loops removes much synthetic padded-capacity overhead but can regress fully active paths;
- the complete-model E12 path has no wholly inactive split at the measured starting points, so the large synthetic guard gain is not the explanation of the complete-model INT4 gap;
- grouping difference-in-differences is not a measurement of one specific HBM/unpack/dequant term.

These measurements characterize the old executor. They **do not establish** that the GQA-grouping crossover is fundamental to compressed attention. The direct dequant-first contraction order may itself create work that a better algebraic factorization removes.

---

## Mathematical contract for the new phase

Before a new optimization is accepted, classify every proposed transformation as one of:

1. **exact equivalence** — identical operator output for all admissible inputs;
2. **conditional exact equivalence** — identical on a declared subspace/structure;
3. **controlled approximation** — explicit deterministic/operator-norm error bound;
4. **probabilistic approximation** — explicit `(epsilon,delta)` or distributional/concentration statement;
5. **heuristic only** — allowed for exploration, not for a claim until converted into one of the above or empirically bounded with clear scope.

Important exact structures already derived include:

- feature-space dual `GL(d)` basis symmetry of `QK^T`;
- orthogonal/Hadamard special case;
- softmax invariance to row-constant logits;
- query-induced quotient `F/ker(Q)`;
- value quotient `T/ker(P)`;
- exact direct-sum feature decomposition before softmax;
- exact token-block softmax merge via `(m,l,o)` sufficient statistics;
- exact movement of group scale after a width-32 code contraction;
- rank-1 correction induced by the current unsigned `q+8` nibble encoding;
- exact V inverse-basis absorption into `W_O` when layout/model compatibility permits.

See `docs/MATH.md` and `docs/OPERATOR_GEOMETRY_AND_CIRCUIT.md`.

---

## E14 — Operator structure probe

E14 is the next primary experiment. Protocol: `experiments/E14_operator_structure/protocol.json`.

E14 is **not a latency experiment**.

### E14A — Query-visible subspace

Measure across multiple layers, KV heads, positions and sequences:

- singular spectrum and numerical rank of each `r x d` GQA query block;
- instantaneous nullity;
- effective/stable rank;
- stacked-window rank for multiple decode windows;
- principal-angle/subspace drift;
- energy captured by 1..8 hardware-aligned 32-wide feature blocks after each candidate exact basis;
- actual score-visible tail error when blocks are omitted diagnostically.

The central gate is whether the large one-step nullity yields a stable low-dimensional/block-structured visible subspace. If it does not, do not build persistent feature-block deletion around it.

### E14B — Basis / quantizer geometry

For identity, current OFD basis, block `H_32`, full `H_256`, and diagnostic alignment bases, measure:

- coherence proxy `||x||_inf/||x||_2`;
- 32-block dynamic range and exponent spread;
- group-scale matrix spectrum/effective rank;
- INT4 reconstruction error;
- **operator-visible** K error `Q Delta K^T C_L`;
- Q integer/radix plane count for exact representation and fixed operator-error budgets.

The key question is whether a basis reduces actual candidate circuit work, not just MSE.

### E14C — V token/output geometry

Measure:

- `N_eff = 1/sum p_t^2`;
- smallest support carrying fixed probability masses;
- dropped-mass error bound;
- actual `||Delta Y||`;
- `||Delta Y W_O||` where available.

Do not mix feature-space nullity and token-support concentration into one claim.

### E14D — Static circuit accounting

For each promising exact/controlled-approximate factorization, count before implementing an optimized kernel:

- packed K/V bytes;
- scale bytes;
- logical 32-wide contractions;
- candidate MMA tiles and M/N/K utilization;
- conversion/representation operations;
- scale/epilogue operations;
- basis/RoPE operations;
- intermediate storage.

The work ledger must be frozen before timing a new microkernel.

---

## Hardware mapping boundary

Verified documentation establishes that PTX exposes `.u4/.s4` MMA families with reduction dimension 32 and that Hopper/CUTLASS supports mixed-input kernel generation involving int4 and f16/bf16.

This does **not** establish that:

- the current Triton kernel lowers to these instructions;
- a source-level int4/f16 expression becomes one native mixed instruction;
- `r=6` maps efficiently to an 8/16-row MMA tile;
- group width 32 is sufficient for speed without considering M/N utilization, packing, registers and epilogues.

Generated PTX/SASS is required before an instruction-level claim.

---

## E15 — Circuit-mapped contraction prototype

E15 begins only if E14 identifies a concrete structural reduction.

Possible prototypes include, depending on E14:

- representation-native signed-s4 32-wide K contraction with post-reduction scale handling;
- mixed-dtype or Q-quantized low-bit contraction chosen from the actual hardware lowering;
- hardware-block feature elision if stable visible-subspace concentration is established;
- V contraction with a mathematically controlled support/representation reduction.

E15 order:

```text
1. freeze algebra and work ledger
2. implement smallest microkernel
3. exact/declared-error numerical gate
4. inspect generated PTX/SASS
5. compare actual bytes/instructions/tile use to prediction
6. only then measure latency
7. only after microkernel success integrate complete model
```

A faster kernel without agreement between algebra, generated work and numerical contract is not a sufficient research result.

---

## Frozen common contract for E00-E13 lineage

Unless an existing experiment explicitly declares otherwise:

- group size = 32;
- page size = 16;
- quantized values use symmetric `absmax/7`, normal finite range `[-7,7]`;
- current packed storage encodes `q+8`, low nibble first;
- scale dtype = FP16;
- FP32 online-softmax / split merge accumulators;
- randomized logical-to-physical page mapping for correctness;
- no dense dequantized KV **cache** materialization;
- historical split/tile/warp/stage settings remain frozen inside their protocols.

A native signed-s4 storage recoding, different quantizer, different basis policy or new circuit-native contraction is a **new lineage** and must not retroactively alter E00-E13 artifacts.

---

## Research growth / stopping rule

Continue when the next step changes understanding of the same objective by resolving:

- an exact equivalence;
- an exploitable null/quotient structure;
- a boundary condition;
- a controlled approximation regime;
- a circuit mapping;
- or a generalizable execution principle.

Stop adding experiments that merely produce another timing table for the same current executor without reducing structural uncertainty.

Branch when the new objective remains meaningful without the compressed-representation/direct-consumption question.

---

## Claim discipline

Do not claim that:

- instantaneous nullity is persistent KV compression;
- Hadamard reduces dimension by itself;
- lower raw reconstruction error implies lower attention/output error;
- E10 grouping preference is universal;
- a 32-wide quantization group automatically implies efficient Tensor Core execution;
- library mixed-dtype support proves one particular native instruction;
- one prompt/layer/head demonstrates stable subspace concentration;
- a static work reduction is a speedup until generated code and timing confirm it;
- a microkernel result is production serving or same-memory-budget throughput.

Novelty must be checked against rotation quantization, low-bit KV kernels, mixed-dtype serving, exact IO-aware attention and current GPU primitives before publication/patent claims.

---

## Required provenance

Every claim-eligible run must preserve:

- exact source SHA and clean/declared worktree state;
- model/checkpoint identity and weights hash where applicable;
- deterministic capture settings or declared randomness/seeds;
- basis definition/hash;
- numerical-rank/effective-rank tolerances;
- raw/sufficient reproducible metrics;
- analysis code SHA;
- generated code for instruction claims;
- separate scientific gate status and execution status.

Later interpretation may supersede an earlier hypothesis; frozen measurements and protocols must remain intact.
