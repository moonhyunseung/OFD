# 05 — Exact symbolic cost reductions

This file counts only algebraically necessary logical work/intermediates. It does **not** claim GPU latency or instruction lowering.

Let

```text
r = query heads per KV head
L = active tokens
G = quantization groups
m = coordinates per group
d = Gm.
```

Current configuration: `r=6`, `G=8`, `m=32`, `d=256`.

## 1. Decode-then-score K semantics

A naive logical path materializes `L*G*m` decoded K values, each requiring scale application after integer decode, then performs `r*L*G*m` query-coordinate products/accumulations.

The exact packed normal form retains the essential code contraction but changes representation work:

```text
R_K[h,t,g] = Σ_j Q[h,g,j] C_K[t,g,j]
```

followed by

```text
Z[h,t] = α Σ_g S_K[t,g] (R_K[h,t,g] - 8 A_Q[h,g]).
```

Logical scale applications attached to dense K coordinates are replaced by scale applications in reduced `(h,t,g)` space. Dense decoded K materialization disappears.

This does **not** reduce the generic bilinear coordinate contraction count `r*L*G*m`; it removes representation reconstruction work and changes where scalar operations occur.

## 2. Decode-then-aggregate V semantics

Naive dense decode materializes `L*G*m` decoded V values.

Exact packed form computes

```text
W[h,t,g] = P[h,t] S_V[t,g]
Y_code[h,g,j] = Σ_t W[h,t,g] C_V[t,g,j]
B[h,g] = 8 Σ_t W[h,t,g]
Y[h,g,j] = Y_code[h,g,j] - B[h,g].
```

Again the generic code contraction remains, but dense V reconstruction and per-coordinate scale work are removed.

## 3. Exact rank factorization is not automatic arithmetic reduction

For `rank(Q)=ρ`, exact factorization

```text
Q = A B
```

requires `ρ` independent functionals in `B`. If `ρ=r` in the generic current case, this does not reduce the number of independent query score channels below r. It may still change reuse/layout, but no dimensional saving is guaranteed.

Similarly, if `rank(P)=r`, exact rank factorization of P does not reduce independent value combinations below r.

## 4. Persistent-nullspace conditional reduction

For a fixed query family with stacked rank `ρ_stack`, the exact observable feature quotient dimension is

```text
d_eff = ρ_stack
```

and persistent invisible dimension is

```text
d - ρ_stack.
```

This is a true exact state-dimension reduction only when the future query family is scoped and the common kernel remains nontrivial. The cost of producing/storing quotient coordinates must be counted separately; rank-nullity alone does not make the quotient free.

## 5. Exact rank-1 scale separation

If

```text
S_K[t,g] = a[t] b[g],
```

then

```text
Z[h,t]
= α a[t] Σ_g b[g] (R_K[h,t,g] - 8 A_Q[h,g]).
```

Token scale is applied once after the group sum; group coefficient can be folded into the query/group-side coefficient. This is an exact algebraic simplification.

For scale rank `s>1`, exact decomposition generally introduces s separable branches; whether that lowers work is not implied by low rank alone.

## 6. What is deliberately not counted here

- Tensor Core tile utilization;
- unpack/convert instruction cost;
- register pressure;
- shared-memory traffic;
- compiler lowering;
- cache/HBM behavior;
- synchronization;
- launch overhead;
- latency.

Those are hardware questions to be considered only after an exact normal form has survived this ledger.