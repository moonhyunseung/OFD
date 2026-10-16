# Kernel design

This file distinguishes the **frozen E00-E13 executor** from the **candidate post-E14 operator-native kernel**. Do not silently mix them.

## 1. Frozen correctness-first executor

Stage 1 grid is based on request / q-head-or-group / split. Each program traverses a context split, resolves logical pages, reads packed INT4 K/V, forms working dequantized values in the inner loop, computes scores and accumulates a stable local softmax/value state.

Stage 2 merges split states `(m,l,o)`.

This executor is the object measured by E10-E12. E13 can time Stage 1 and merge independently as a diagnostic.

### Known properties / limitations

- packed KV cache is consumed directly; no dense full-cache dequantized materialization is allowed;
- the inner loop still follows a dequant-first working-value order;
- GQA grouping amortizes some work of that order;
- the current unsigned `q+8` encoding induces nibble decoding / representation correction;
- current query/output transforms are not yet the final operator-placed design;
- E12 proves that a small control-flow change can alter active-path performance, so source edits are not mechanism proofs.

The E10 grouping crossover is therefore a property of this executor under the tested conditions, not yet a general law.

## 2. Candidate post-E14 design target

The new target starts from exact packed-code algebra.

For one 32-wide K group using current unsigned nibble storage:

```text
Q_g K_g^T
= (Q_g U_g^T) D(s_g)
  - 8 (Q_g 1) s_g^T.
```

For V:

```text
P V_g
= (P D(s_g)) U_g
  - 8 (P s_g) 1^T.
```

The desired kernel therefore does **not** reconstruct an `L x 32` float K/V group and then contract it. It performs a representation-native code contraction and applies scale/correction in the reduced result space.

A future signed-s4 storage recoding may remove the rank-1 unsigned-code correction entirely, but that is a new storage lineage and must pass independent numerical/storage compatibility gates.

## 3. Candidate dataflow

```text
Q / RoPE / legal basis placement
        ↓
hardware-aligned 32-wide Q blocks
        ↓
representation-native K-code contraction
        ↓
post-reduction K scale / exact representation correction
        ↓
sum feature-group score contributions
        ↓
logits
        ↓
online softmax sufficient state
        ↓
probability coefficient × V-scale preparation
        ↓
representation-native V-code contraction
        ↓
output in chosen V basis
        ↓
optional inverse basis absorbed into W_O when exact/legal
```

## 4. Feature and token decomposition rules

Feature groups may be computed independently before softmax because

```text
QK^T = sum_g Q_g K_g^T.
```

They must be recombined before the softmax nonlinearity.

Token/page blocks may be computed independently only through the exact stable-softmax sufficient state `(m,l,o)` and its associative merge.

GQA heads share K/V and should be treated as a matrix/reuse dimension rather than as independent reasons to decode the same representation repeatedly.

## 5. Hardware mapping on H200

The current group width 32 is potentially useful because PTX provides `.s4/.u4` MMA families with `K=32`, and CUTLASS documents Hopper mixed-input int4/f16/bf16 kernel families.

However, a valid OFD hardware claim requires all of the following:

1. exact logical matrix shape declared, e.g. `(r x 32) @ (32 x N)`;
2. actual source storage interpretation declared;
3. M/N/K tile utilization calculated (`r=6` is not naturally an 8/16-row tile);
4. conversion/scale/zero-point work counted;
5. registers/shared-memory/intermediate requirements accounted for;
6. generated PTX/SASS archived;
7. numerical contract passed;
8. measured timing agrees with the predicted work reduction.

Do not infer Tensor Core use or a native mixed-dtype instruction solely from high-level source types.

## 6. Basis placement requirements

A basis transform is allowed in a hot path only after its location is justified algebraically.

- Across a bilinear pairing, use adjoint/dual transfer.
- Across RoPE, analyze the conjugated family `A^{-1} Omega_t A`; commutation is only the easiest special case.
- Before quantization, basis choice can change quantizer geometry because quantization is nonlinear.
- On V, an inverse transform may be absorbed into `W_O` exactly when model/head layout permits.

Hadamard is an exact orthogonal basis change, not a dimension reduction operation.

## 7. Dimension reduction requirements

Instantaneous `rank(Q)<=r` gives a large exact nullspace, but a new kernel may exploit feature-block elimination only if E14 establishes a stable/effective visible subspace or a controlled approximation bound across representative decode states.

For block projector `P_k`, the basic score-tail bound is

```text
|| Q(I-P_k)K^T ||_F
<= ||Q(I-P_k)||_F ||K||_2.
```

No persistent K block may be skipped solely because one sampled Q has zero/small energy there.

V token reduction likewise requires declared dropped probability mass and output-space error/bound.

## 8. E15 implementation rule

Do not start an optimized post-E14 kernel until E14 produces a concrete reduction and a frozen static work ledger.

E15 sequence:

```text
freeze algebra
-> freeze byte/op/MMA/tile ledger
-> implement smallest representative microkernel
-> numerical gate
-> generated-code inspection
-> measured work confirmation
-> latency confirmation
-> complete-model integration only after the above
```

An optimization that is faster but cannot be reconciled with the declared operator/circuit model is engineering evidence, not yet the desired research result.

## 9. Legacy v2.3 timing boundary

The existing runtime still exposes the old conceptual levels:

- `triton_paged_decode_core(...)`: frozen packed-INT4 Stage 1 + Stage 2 core;
- `triton_paged_decode(...)`: wrapper including the current transform boundary.

These remain valid for historical E00-E13 reproduction. The new operator-native path must use new symbols/files/experiment lineage rather than changing the meaning of historical entry points.
