# 02 — Exact quantized normal form

The objective is to express the **same decoded quantized attention** without materializing dense decoded K/V.

## 1. Frozen decoded representation

For token `t`, group `g`, coordinate `j`:

```text
Khat[t,g,j] = S_K[t,g] (C_K[t,g,j] - 8)
Vhat[t,g,j] = S_V[t,g] (C_V[t,g,j] - 8)
```

where `C_*` are stored unsigned nibbles.

## 2. K-side exact normal form

For query head `h`, token `t`:

```text
Z[h,t]
= α Σ_g Σ_j Q[h,g,j] S_K[t,g] (C_K[t,g,j] - 8)
```

By distributivity:

```text
Z[h,t]
= α Σ_g S_K[t,g]
    ( Σ_j Q[h,g,j] C_K[t,g,j]
      - 8 Σ_j Q[h,g,j] ).
```

Define

```text
R_K[h,t,g] = Σ_j Q[h,g,j] C_K[t,g,j]
A_Q[h,g]   = Σ_j Q[h,g,j].
```

Then

```text
Z[h,t] = α Σ_g S_K[t,g] (R_K[h,t,g] - 8 A_Q[h,g]).
```

### Exact consequences

Relative to decode-then-dot semantics:

- no dense `Khat[t,g,j]` tensor is mathematically required;
- the scale multiply occurs after the width-32 coordinate reduction;
- subtraction of `8` need not occur per coordinate;
- zero-point correction is one scalar term per `(h,t,g)` after precomputing `A_Q[h,g]` for the current Q.

These are identities, not empirical hypotheses.

## 3. V-side exact normal form

For attention probability `P[h,t]`:

```text
Y[h,g,j]
= Σ_t P[h,t] S_V[t,g] (C_V[t,g,j] - 8).
```

Define

```text
W[h,t,g] = P[h,t] S_V[t,g].
```

Then

```text
Y[h,g,j]
= Σ_t W[h,t,g] C_V[t,g,j]
  - 8 Σ_t W[h,t,g].
```

### Exact consequences

- no dense `Vhat` tensor is mathematically required;
- scale is applied to the reduced probability coefficient rather than to all 32 decoded coordinates;
- zero-point correction is scalar in `(h,g)` and broadcast across the 32 coordinates.

## 4. Packed-pair decomposition

Each byte contains two nibbles. After decoding nibble values logically as even/odd code matrices `C_e,C_o`, and splitting query coordinates `Q_e,Q_o`,

```text
Q C^T = Q_e C_e^T + Q_o C_o^T.
```

This is exact and permits one packed byte to supply two code coordinates without first constructing a dense dequantized vector.

## 5. Bit-plane decomposition

For unsigned 4-bit code

```text
C = B0 + 2 B1 + 4 B2 + 8 B3,
Bk ∈ {0,1}.
```

Therefore

```text
Q C^T = Q B0^T + 2 Q B1^T + 4 Q B2^T + 8 Q B3^T.
```

Again this is exact. Whether packed-pair or bit-plane form is cheaper is a hardware-lowering question, not an algebra question.

## 6. Exact scale separability condition

Let `S ∈ R^(L×G)`. If and only if `rank(S)=1` for nonzero `S`, there exist vectors `a,b` such that

```text
S[t,g] = a[t] b[g].
```

Then K-side group dependence `b[g]` may be absorbed into the query/group coefficient and token dependence `a[t]` applied after the group contraction, exactly.

If `rank(S)>1`, an exact rank decomposition requires that many separable terms in general. A numerically small tail singular spectrum is an approximation question and is not part of this exact result.