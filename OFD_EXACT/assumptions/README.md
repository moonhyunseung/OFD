# Assumptions and boundary questions

Experiments in `OFD_EXACT` exist to locate the boundary of a theorem's assumptions, not to discover algebra that is already known.

Each assumption must be written as:

```text
Claim enabled by theorem:
Required assumption:
Exact failure condition:
Boundary variable:
Minimal observation needed:
What result terminates the branch:
```

## A1 — Persistent common nullspace

### Theorem-enabled claim
A feature subspace may be discarded persistently and exactly for a scoped future-query family if it lies in

```text
N = ∩_t ker(Q_t).
```

### Boundary question
For the stated model/layer/head/query scope, when does the stacked query map reach full feature rank?

```text
Q_stack(w) = [Q_1; ...; Q_w]
nullity(w) = d - rank(Q_stack(w)).
```

### Minimal experiment
Measure exact/algebraic rank under an explicitly stated numerical representation only because the actual model-generated Q family is not known symbolically in advance.

### Termination
If the scoped stacked map reaches rank d, exact persistent feature deletion through this nullspace route is closed for that scope.

## A2 — Exact scale separability

### Theorem-enabled claim
If `rank(S)=1`, token and group scale dependence separates exactly.

### Boundary question
For which layer/head/context states, if any, is the stored scale matrix exactly rank one?

### Minimal experiment
Evaluate exact matrix rank under the frozen stored FP16 scales. Do not substitute stable rank or spectral decay for exact rank in the exact branch.

### Termination
If rank exceeds one generically, the exact rank-one branch closes. Numerical low-rank approximation is a separate project.

## A3 — RoPE commutant / normalizer

### Theorem-enabled claim
A transform may cross RoPE for free only if it commutes with every scoped RoPE operator. It may cross with exact rewritten RoPE if conjugation preserves an acceptable structured family.

### Boundary question
Characterize the commutant and, separately, transforms whose conjugation keeps RoPE cheap/structured.

This is primarily symbolic algebra. Empirical testing is unnecessary until a hardware cost is attached to a proven structured conjugate.

## A4 — Dynamic gate commutant

### Theorem-enabled claim
A V-side basis may cross the dynamic diagonal gate unchanged only within common equal-gate coordinate classes.

### Boundary question
Do nontrivial coordinate equivalence classes exist over the intended scope?

### Minimal observation
Check exact equality structure of captured/stored gate trajectories only after the symbolic commutant condition is written.

### Termination
If all coordinate gate trajectories are distinct over scope, dense cross-coordinate mixing cannot cross the gate for free.

## A5 — Hardware benefit of an exact normal form

### Theorem-enabled claim
K1/V1 eliminate dense dequantized intermediates algebraically.

### Boundary question
On a specific target circuit, at what workload does the native packed realization stop outperforming the frozen decode-first executor?

This is a hardware crossover experiment, not an algebra experiment. Its variables may include context length, batch, GQA ratio, tile shape, and occupancy, but only after native instruction lowering is verified.

## Experimental rule

A valid experiment must be traceable to one unresolved assumption or boundary above. If the answer follows from an exact theorem/invariant already in `algebra/`, do not run the experiment.