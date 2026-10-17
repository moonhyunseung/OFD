# Boundary experiments only

No exploratory performance sweep belongs here unless it tests a named assumption from `../assumptions/README.md`.

## Required protocol template

Every experiment must declare:

1. **Exact theorem/identity already established.**
2. **Single unresolved assumption.**
3. **Failure condition.**
4. **Boundary variable(s).**
5. **Minimal measurement needed to decide the assumption.**
6. **Stopping rule.**
7. **Why the measurement cannot be replaced by algebra alone.**

## Forbidden experiment types in the exact branch

- testing whether an orthogonal transform changes rank;
- testing whether block-Hadamard changes the norm of the same block;
- searching arbitrary basis transforms for a lower-rank exact representation;
- using truncated SVD/block dropping/token dropping while labeling the result exact;
- running latency sweeps before a mathematically equivalent target normal form is fixed;
- treating profiler attribution as proof of an algebraic cause.

## Allowed examples

- find the smallest query window at which a scoped common nullspace vanishes;
- test whether stored scale matrices are exactly rank one anywhere under the frozen representation;
- verify whether a proposed exact packed contraction lowers to the intended H200 primitive;
- find the context/batch crossover where an already-proved exact packed implementation loses its hardware advantage.

Historical E00-E14 results are predecessor evidence and are not copied here.