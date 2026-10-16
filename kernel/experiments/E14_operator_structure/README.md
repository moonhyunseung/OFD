# E14 — Operator Structure Probe

E14 is the first experiment in the post-E12 operator/circuit design phase.

It is **not** a latency sweep and **not** a hot-kernel optimization experiment. Its purpose is to determine whether the exact quotient-space and block-structured algebra in `docs/OPERATOR_GEOMETRY_AND_CIRCUIT.md` exposes a computational reduction that is present in real model states.

## Scientific question

> Which mathematically available reductions are actually exploitable in the target model distribution, and do they align with 32-wide low-bit hardware blocks strongly enough to justify a new kernel lineage?

## Frozen non-goals

E14 must not:

- tune Triton warps/stages/split size;
- change the existing E10-E12 kernel and then infer causality from latency;
- claim persistent KV deletion from one-step nullity;
- claim Hadamard reduces dimension by itself;
- optimize a basis against downstream task accuracy without first measuring the operator geometry;
- use timing as the primary acceptance criterion.

## E14A — Query-visible subspace

For each sampled layer / KV head / decode step, form the GQA query block

```text
Q_t in R^(r x d),  r=Hq/Hkv.
```

Record:

1. exact numerical rank under declared tolerance;
2. singular values;
3. stable/effective rank metrics;
4. instantaneous nullity `d-rank(Q_t)`;
5. window-stacked rank for windows `w`;
6. principal angles between dominant subspaces at adjacent / separated steps;
7. subspace drift versus layer and context.

The primary question is whether the large instantaneous nullity remains useful over a nontrivial window.

### 32-wide block concentration

For each candidate **exact** basis `A`, compute `Q_t A` and partition feature space into eight 32-wide blocks.

For `k=1..8`, record the best or policy-constrained captured energy

```text
capture_k = ||Q_t A P_k||_F^2 / ||Q_t A||_F^2.
```

Also record score-visible tail error on the actual K sample:

```text
Delta Z_k = Q_t A (I-P_k) (K A^{-T})^T.
```

Do not infer computational savings from energy alone; pair it with operator-visible score error.

Candidate basis classes should include at least:

- identity;
- the current OFD orthogonal/Hadamard basis where defined;
- `H_32` blockwise mixing;
- full `H_256` mixing;
- a data-derived orthogonal alignment basis used only as an upper-bound diagnostic unless it can be made runtime/legal.

## E14B — Quantizer/basis geometry

For K, V, and Q blocks, measure before/after each candidate basis:

- `||x||_inf / ||x||_2` coherence proxy;
- per-32-block max/range statistics;
- per-block exponent/dynamic-range spread;
- group-scale matrix singular values;
- scale-matrix effective rank;
- current INT4 reconstruction error;
- operator-visible K error `Q Delta K^T C_L`;
- operator-visible V error `P Delta V` and, where available, `P Delta V W_O`.

### Integer/radix plane diagnostic

For Q blocks, determine the number of signed integer/radix planes needed for:

1. exact finite-representation reconstruction under the chosen decomposition;
2. fixed operator-error budgets.

The purpose is to test whether basis flattening translates into **fewer candidate MMA/conversion stages**, not merely lower raw MSE.

## E14C — V token/output geometry

For each attention probability row, record:

```text
N_eff = 1 / sum_t p_t^2
```

and the minimum support sizes carrying fixed masses such as `0.90, 0.95, 0.99, 0.999`.

For controlled support truncation, record:

- dropped probability mass `delta`;
- norm bound `delta * max_t ||v_t||`;
- actual `||Delta Y||`;
- actual `||Delta Y W_O||` where available.

This axis is separate from feature-space rank/nullity.

## E14D — Circuit feasibility accounting

No optimized kernel is required. For each mathematically promising factorization, produce a static work ledger:

```text
packed K bytes read
packed V bytes read
scale bytes read
logical 32-wide contractions
candidate MMA tile count
M/N/K utilization
conversion count/model
scale/epilogue op count
basis/RoPE transform count
expected intermediate storage
```

On H200, compare the dimensions against documented PTX/CUTLASS Hopper families. If a prototype is compiled, archive generated PTX/SASS and state exactly what primitive was emitted; do not infer it from source types.

## Sampling discipline

The probe must cover enough variation to distinguish structural stability from a cherry-picked state:

- multiple layers spanning early/middle/late depth;
- multiple KV heads;
- multiple decode positions / context regimes;
- multiple sequences/prompts;
- fixed model weights and deterministic capture settings for claim-eligible runs.

A smoke run may use a much smaller subset, but smoke results are not structural claims.

## Required artifacts

Each claim-eligible E14 run must archive:

- `MANIFEST.json` with exact model/checkpoint/source SHA/environment;
- raw Q/K/V/P-derived metrics or sufficient reproducible summaries;
- declared numerical-rank/effective-rank tolerances;
- basis definitions and hashes;
- operator-error tables;
- 32-block concentration tables;
- V support tables;
- static circuit-work ledger;
- analysis script SHA;
- no latency claim unless a separately versioned prototype is explicitly part of the run.

## Decision gates

### Gate A — persistent feature structure

Proceed to feature-block-elision design only if a stable basis captures a useful fraction of query-visible energy in fewer than eight 32-wide blocks with acceptably bounded/observed operator error across the declared sample set.

### Gate B — basis-to-circuit benefit

Proceed to Q integer/radix contraction design only if the basis materially reduces the plane/conversion requirement at fixed operator error, or a documented mixed-dtype path already removes the need for such conversion.

### Gate C — V token reduction

Proceed to token-support approximation only if probability concentration is stable enough that a declared `delta` yields acceptable output-space error across the sample set.

### Gate D — new kernel lineage

A new hot kernel begins only after at least one of Gates A–C identifies a concrete reduction and the static circuit ledger predicts fewer bytes/instructions/MMA tiles than the current executor.

## Interpretation discipline

Negative results are useful:

- rapidly rotating query subspaces imply that instantaneous nullity is not persistently exploitable;
- weak 32-block concentration rejects hardware-block feature elision for that basis/model;
- Hadamard flattening without plane/count reduction rejects a circuit-level benefit despite better coherence;
- diffuse V probabilities reject aggressive support truncation.

E14 is successful if it sharply narrows the design space, even if it blocks a proposed optimization.
