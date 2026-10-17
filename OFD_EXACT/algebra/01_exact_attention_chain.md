# 01 — Exact attention chain and local equivalence

## 1. Local proof rule

For an original edge

```text
X_i --f_i--> X_(i+1)
```

and transformed representations

```text
X_i --φ_i--> X~_i
X_(i+1) --φ_(i+1)--> X~_(i+1),
```

an exact replacement `f~_i` must satisfy the commuting relation

```text
φ_(i+1) ∘ f_i = f~_i ∘ φ_i.
```

If `φ_i` is invertible, then the canonical transformed operator is

```text
f~_i = φ_(i+1) ∘ f_i ∘ φ_i^(-1).
```

If no such exact relation is established, the transformed state must be reconstructed before passing to the original next operator.

## 2. Exact representation change

For any invertible `A ∈ GL(d)`, a feature representation may be changed without information loss. However, downstream maps must be transformed compatibly.

For the score bilinear form, dual action gives

```text
(Q A) (K A^{-T})^T = Q K^T.
```

For orthogonal `H`, `H^{-T}=H`, so

```text
(Q H)(K H)^T = Q K^T.
```

This is an exact change of coordinates, not a dimensional reduction.

## 3. Quotient factorization

A non-invertible map is exact only when lost distinctions are invisible to the scoped downstream observable.

For fixed `Q`, define

```text
k1 ~_Q k2  iff  Q k1 = Q k2.
```

Then equivalence classes are cosets of `ker(Q)` and the first isomorphism theorem gives

```text
F / ker(Q) ≅ im(Q).
```

The score map factors exactly through this quotient.

A quotient may be used as a persistent representation only if every future required map also factors through the same quotient. For a family `{Q_t}`, the exact persistent invisible subspace is

```text
N_persist = ∩_t ker(Q_t)
          = ker([Q_1; Q_2; ...]).
```

Only directions in `N_persist` may be discarded without reconstruction for all queries in that scope.

## 4. Softmax quotient

Row-wise softmax is invariant to an additive row constant:

```text
softmax(z + c 1) = softmax(z).
```

Therefore logits are observed only modulo `span{1}` along the token axis. Let

```text
C_L = I - (1/L) 11^T.
```

Then centered logits capture the exact softmax-relevant quotient:

```text
Z ~ Z C_L.
```

For K-side perturbation `ΔK`, exact softmax invisibility is guaranteed when

```text
Q ΔK^T C_L = 0.
```

## 5. Exact token partition through softmax

Feature contributions must be summed before softmax because, in general,

```text
softmax(Z1 + Z2) != softmax(Z1) + softmax(Z2).
```

Token partitions are different: each token block can be summarized by stable online-softmax sufficient statistics `(m,l,o)` and those summaries admit an exact associative merge. This is an exact factorization of the token reduction, not an approximation.

## 6. SVD / rank factorization rule

A full compact rank factorization is exact. If `rank(Q)=ρ`, there exist

```text
Q = A B,
A ∈ R^(r×ρ),
B ∈ R^(ρ×d),
```

and therefore

```text
Q K^T = A (B K^T).
```

No exact factorization through an inner dimension `< ρ` exists.

Consequently truncated SVD (`k < ρ`) is not an exact executor rewrite unless the truncated directions are independently proved to belong to a downstream-invisible quotient. It is excluded from the exact path otherwise.