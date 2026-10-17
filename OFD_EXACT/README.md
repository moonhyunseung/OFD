# OFD_EXACT

`OFD_EXACT` restarts OFD from exact algebra before hardware implementation.

## Frozen predecessor

- repository: `moonhyunseung/OFD`
- predecessor branch: `ofd-v2.3-measurement-safe`
- frozen predecessor commit: `58bbec23ba1c862f98b71af014c4a56e9a9e7a88`
- predecessor scope: E00-E14 measurement and exploratory analysis history

Nothing under the predecessor branch is reinterpreted as an exact-algebra result. E00-E14 remain historical measurements and diagnostics.

## Research order

The order is mandatory:

1. **Algebra** — define spaces, maps, representations, exact identities, kernels, images, quotients, commutants, and factorization constraints.
2. **Equivalence** — every proposed rewrite must preserve the represented state bijectively, factor through an exact downstream quotient, or provide an explicit reconstruction/conjugation map.
3. **Assumptions** — isolate only those facts not supplied by algebra alone.
4. **Boundary experiments** — experiments test where an assumption fails; they do not substitute for a theorem.
5. **Hardware mapping** — among mathematically equivalent normal forms, choose those that match available circuit primitives.
6. **Implementation** — code is written only after the algebraic contract and hardware lowering target are fixed.

## Exact-path rule

A candidate transformation may enter the exact path only if one of the following is written down and proved:

- pointwise equality of maps;
- an invertible representation change with an explicit inverse;
- a commuting diagram with a transformed downstream map;
- factorization through a quotient whose equivalence classes are invisible to every downstream observable under the stated scope.

Approximate operations such as truncated SVD, block dropping, token dropping, numerical low-rank replacement, or approximate basis persistence are not part of the exact path. They may be studied later in a separate approximation branch.

## Primary semantic reference

Quantization itself is lossy relative to the original FP tensor. Therefore exact executor work is referenced to the **frozen decoded quantized representation**, not to the pre-quantized FP tensor.

For a packed code/scale object `c`, decoder `D`, and downstream operator `F`, an exact representation-native replacement `F_tilde` must satisfy

```text
F_tilde(c) = F(D(c))
```

for every admissible packed state under the frozen representation contract.

## Current algebraic priority

The first exact targets are:

- remove dense K dequantization by contracting directly with packed codes and applying scale / zero-point correction after the coordinate reduction;
- remove dense V dequantization by moving V scale into probability coefficients and applying zero-point correction as a broadcast term;
- classify every transform by exact equality, isomorphism, quotient factorization, or approximation;
- identify exact conditional reductions such as persistent common nullspaces or exact scale separability;
- reject transformations whose benefit is already ruled out by invariance theorems.

See `algebra/` and `assumptions/`.