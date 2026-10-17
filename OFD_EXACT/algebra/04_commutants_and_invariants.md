# 04 — Commutants, invariants, and transformations that cannot deliver the claimed benefit

The purpose of this file is to eliminate impossible or structurally poor candidates before experiments.

## 1. Orthogonal right actions

For orthogonal `H`:

```text
rank(QH) = rank(Q)
σ_i(QH) = σ_i(Q)
||xH||_2 = ||x||_2.
```

Therefore orthogonal rotation cannot by itself:

- reduce exact rank;
- reduce nullity complement dimension;
- reduce Euclidean energy;
- create exact dimensional compression.

Its legitimate roles are coordinate redistribution, coherence control, quantizer conditioning, or primitive/layout alignment.

## 2. Block Hadamard invariant

For

```text
H_block = I_G ⊗ H_m,
```

each pre-existing group norm is invariant:

```text
||x_g H_m||_2 = ||x_g||_2.
```

Hence block-Hadamard cannot move energy from one quantization group to another. Any experiment whose hypothesis is "H32 concentrates energy into fewer existing 32-wide groups" is invalid because the answer is fixed by the theorem.

## 3. RoPE mobility

Let position-dependent RoPE operator be `R_p`. A basis `A` crosses RoPE unchanged iff

```text
A R_p = R_p A  for all p.
```

That is, `A` must lie in the common commutant

```text
Comm({R_p}).
```

For general invertible A, exact movement is still possible only by replacing RoPE with

```text
R'_p = A^-1 R_p A.
```

Thus the algebraic question is not whether a transform can be moved in principle, but whether it is in the commutant or normalizes RoPE into another cheap structured family.

## 4. Dynamic diagonal gate mobility

For a time-dependent diagonal gate

```text
D_t = diag(g_t),
```

a basis A crosses without changing the gate iff

```text
A D_t = D_t A  for all t.
```

Entrywise,

```text
A_ij (g_t,j - g_t,i) = 0  for all t.
```

Therefore coordinates i and j may mix freely only if their gate trajectories are identical over the stated scope. With generic distinct trajectories the common commutant collapses toward diagonal/block-diagonal structure.

## 5. Quantization breaks full linear symmetry

For a nonlinear groupwise quantizer `Q4`, in general

```text
Q4(xA) != Q4(x) A.
```

Therefore an exact FP basis symmetry does not imply equivalence after requantization. A transform may still be useful as a quantizer conditioner, but that belongs to representation design and must not be confused with exact executor equivalence for a fixed packed state.

## 6. Rank lower bound on exact factorization

If a linear map `M` has rank ρ, any exact factorization

```text
M = A B
```

through an inner vector space of dimension k requires

```text
k >= ρ.
```

Thus full-rank factorization can reorganize work but cannot cross the rank lower bound. Numerical effective rank matters only after an approximation budget is introduced.