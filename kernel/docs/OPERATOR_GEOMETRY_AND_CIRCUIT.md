# OFD Operator Geometry and Circuit Map

Status: **working mathematical design note, not a validated performance claim**.

This document reframes the next OFD phase. E10–E12 are retained as measurements of the current correctness-first packed-INT4 executor. They should not be used to infer that the current `g` crossover is a fundamental property of compressed attention. The next design phase starts from the attention operator, its exact equivalence classes, the nonlinear boundaries that prevent motion of operators, and the matrix primitives that Hopper can execute efficiently.

The central question is now:

> **Can the exact or controlled-approximate attention operator be factored directly over a compressed representation so that unnecessary dimensions, reconstruction work, and representation conversion disappear before kernel tuning begins?**

The intended reasoning order is:

```text
operator/function
  -> domain/codomain
  -> kernel/image/quotient
  -> exact symmetries and movable operators
  -> exact decompositions
  -> controlled approximations and error operators
  -> hardware-native contraction geometry
  -> only then: kernel implementation and timing
```

---

## 1. Spaces and operators

For one KV head and the `r` query heads that share it, define:

- feature space: `F = R^d`
- token space: `T = R^L`
- GQA query-head space: `H = R^r`

For the current Qwen geometry, `d=256`, `r=6`.

Using row-vector/matrix notation:

- `Q in Hom(F,H)`, represented by an `r x d` matrix;
- `K,V in Hom(F,T)`, represented by `L x d` matrices;
- `K* in Hom(T,F)` is the Euclidean adjoint, represented by `K^T`.

The score map is the bilinear composition

```text
B(Q,K) = Q o K* = Q K^T in Hom(T,H).
```

Row-wise softmax maps logits into a product of simplices:

```text
S : Hom(T,H) -> (Delta^(L-1))^r.
```

For `P=S(B(Q,K))`, value aggregation is

```text
C(P,V) = P o V = P V in Hom(F,H).
```

Therefore attention is the composition

```text
A(Q,K,V) = C(S(B(Q,K)), V).
```

This factorization, rather than a particular Triton implementation, is the object to optimize.

---

## 2. Exact feature-space quotient induced by Q

Fix one decode step and one KV head group. Let

```text
rho_Q = rank(Q) <= r.
```

For a key vector `k in F`, only `Qk` is observable by the score map. If `n in ker(Q)`, then

```text
Q(k+n) = Qk.
```

Thus keys are equivalent modulo `ker(Q)`:

```text
k1 ~_Q k2  iff  k1-k2 in ker(Q).
```

The score map therefore factors through the quotient

```text
F / ker(Q) ~= im(Q)
```

with

```text
dim(F / ker(Q)) = rho_Q,
nullity(Q) = d-rho_Q.
```

For `d=256`, `r=6`, one decode step has `rho_Q <= 6`; if full row rank, the instantaneous feature nullity is 250. This does **not** imply that 250 dimensions of the persistent KV cache can be deleted: `ker(Q)` changes with the query. It says that the current score operator observes only a low-dimensional quotient of feature space.

A persistent computational reduction is possible only if the visible subspace is stable or approximately stable across a useful decode window/layer population. That is an empirical structural question, not a latency question.

---

## 3. Softmax adds an exact token-space quotient

For each query row,

```text
softmax(z + c 1) = softmax(z).
```

Let

```text
C_L = I - (1/L) 1 1^T.
```

Then

```text
softmax(Q K^T) = softmax(Q K^T C_L).
```

For fixed `Q`, define the linear pre-softmax observable map

```text
L_Q(K) = Q K^T C_L.
```

Its rank is

```text
rank(L_Q) = rho_Q (L-1),
```

and its nullity over the `Ld`-dimensional K space is

```text
nullity(L_Q) = Ld - rho_Q(L-1).
```

For large `L`, the fraction of K degrees of freedom visible to this one-step operator approaches `rho_Q/d`.

The correct interpretation is an operator quotient, not a storage claim.

---

## 4. V has a different quotient

For fixed attention probabilities `P in R^(r x L)`, value aggregation is linear:

```text
M_P(V) = P V.
```

If `pi = rank(P)`, then

```text
rank(M_P) = pi d,
nullity(M_P) = (L-pi)d.
```

Thus V is observed only through `T / ker(P)` in the token axis. Again, the quotient changes with `P`; it cannot be converted into persistent compression without stability or a controlled approximation argument.

For final-output error, the more relevant quantity is often not `||Delta V||` but

```text
|| P Delta V W_O ||,
```

because directions annihilated by later operators are functionally invisible.

---

## 5. Exact feature-space symmetry: the basis is not unique

For any invertible `A in GL(F)`, define

```text
Q_A = Q A,
K_A = K A^{-T}.
```

Then

```text
Q_A K_A^T = Q A A^{-1} K^T = Q K^T.
```

Therefore full-precision score computation has a `GL(d)` dual-action symmetry. Orthogonal rotation is the well-conditioned special case `A^{-T}=A`.

For values, if `V_B = V B`, then

```text
P V = (P V_B) B^{-1}.
```

The basis is therefore a design variable. The correct optimization question is not “should OFD use Hadamard?” but:

> **Which representative of the exact basis-equivalence class gives the simplest compressed representation and the cheapest downstream operator graph?**

---

## 6. Why Hadamard is valid, and what it does not do

A normalized Hadamard transform `H` is an orthogonal automorphism of feature space:

```text
H H^T = I.
```

Thus

```text
(QH)(KH)^T = QK^T,
(PVH)H^T = PV.
```

Hadamard does **not** create nullity and does **not** reduce dimension by itself. It rotates the visible and null subspaces while preserving rank, singular values, and Euclidean norms.

Its relevance to quantization is incoherence/flattening: randomized Hadamard transforms can spread coordinate energy and reduce sensitivity to coordinate outliers. QuIP# uses randomized Hadamard incoherence processing for low-bit quantization, QuaRot exploits exact rotation invariances for end-to-end 4-bit inference, and recent theory gives provable quantization guarantees for randomized Hadamard transforms.

For `d=256=8*32`, the Sylvester Hadamard admits, up to ordering/sign convention,

```text
H_256 = H_8 tensor H_32.
```

This separates two possible roles:

- `H_32`: mix coordinates within one 32-wide quantization/hardware block;
- `H_8`: mix energy across the eight blocks.

Block-only mixing cannot remove imbalance across blocks. Full mixing can, but it may interact differently with RoPE and with block sparsity/alignment objectives.

---

## 7. Operator placement: commuting is sufficient, conjugation is general

Let `Omega_t` denote the position-dependent RoPE operator on feature space. Moving a basis transform `A` across RoPE gives

```text
Omega_t A = A (A^{-1} Omega_t A).
```

Commutation `A Omega_t = Omega_t A` is a sufficient easy case, not the only valid case. The general placement question is:

```text
Does A^{-1} Omega_t A remain in a cheap structured operator family?
```

A transform that improves quantization but conjugates RoPE into a dense expensive operator may be a net loss. RotateKV is relevant prior art because it explicitly addresses rotation placement relative to RoPE.

For V, there is no RoPE barrier. If V is stored as `V B`, then the inverse can be moved into the output projection:

```text
(P V B) (B^{-1} W_O) = P V W_O.
```

Therefore a V-side inverse transform can, in principle, disappear from the runtime graph by changing the following weight matrix. This is exact algebra; implementation/model-weight compatibility must still be checked.

---

## 8. Quantization is a nonlinear boundary that breaks basis symmetry

Let the quantizer be an encoder/decoder pair

```text
E : Hom(F,T) -> R_q,
D : R_q -> Hom(F,T),
```

where `R_q` is the packed-code/scale representation space.

Generally,

```text
D(E(KA)) != D(E(K)) A.
```

Thus quantization breaks the full-precision basis symmetry. The role of basis selection is to choose a representative of an exact full-precision equivalence class that is favorable for the nonlinear encoder and subsequent hardware mapping.

The OFD target should therefore be a representation-native contraction

```text
B_tilde(Q, c_K) = B(Q, D(c_K))
```

without materializing `D(c_K)` as a dense K matrix, and analogously

```text
C_tilde(P, c_V) = C(P, D(c_V)).
```

A direct-consume kernel is mathematically successful when these diagrams commute; performance is a separate implementation question.

---

## 9. Exact groupwise INT4 contraction without dense dequantization

Current storage uses symmetric quantized values `q in [-7,7]`, packed as unsigned nibbles `U=q+8`, with one FP16 scale per 32-value group.

Let one K group be

```text
K_g = D(s_g^K) (U_g^K - 8 * 1_L 1_m^T),
m=32.
```

For `Q_g in R^(r x m)`, exact score contribution is

```text
Q_g K_g^T
= (Q_g U_g^{K,T}) D(s_g^K)
  - 8 (Q_g 1_m) (s_g^K)^T.
```

This exposes two exact facts:

1. scale application can occur **after** the width-32 contraction rather than on every reconstructed K element;
2. the unsigned-zero-point correction is a rank-1 outer product, not a dense elementwise operation.

For V,

```text
V_g = D(s_g^V) (U_g^V - 8 * 1_L 1_m^T)
```

and

```text
P V_g
= (P D(s_g^V)) U_g^V
  - 8 (P s_g^V) 1_m^T.
```

Again, the dense dequantized `L x m` V block is unnecessary.

A future storage-lineage change may recode the same `[-7,7]` values directly as signed two's-complement s4 nibbles. If the hardware path consumes signed s4, the `+8/-8` representation correction can disappear entirely. This is an exact recoding of stored codes, but it is **not** the current frozen storage contract and must be versioned separately.

---

## 10. Exact packed decomposition

Each packed byte contains two 4-bit coordinates. Splitting a group into even/odd coordinates gives

```text
U_g = interleave(L_g, H_g),
Q_g = [Q_g^even, Q_g^odd],
```

so

```text
Q_g U_g^T
= Q_g^even L_g^T + Q_g^odd H_g^T.
```

No float K block is required.

A nibble can also be decomposed into bit planes

```text
U = B0 + 2 B1 + 4 B2 + 8 B3,
```

hence

```text
Q U^T
= Q B0^T + 2 Q B1^T + 4 Q B2^T + 8 Q B3^T.
```

This is exact algebra. Whether bit-plane execution is faster is purely a hardware-mapping question.

---

## 11. Tensor/direct-sum structure

With `d=256`, group width `m=32`, group count `G=8`, the chosen coordinate representation can be written

```text
F ~= G_space tensor M_space,
dim(G_space)=8,
dim(M_space)=32.
```

Paged token space is a direct sum

```text
T = direct_sum_p T_p,
dim(T_p)=page_size.
```

Feature/group contributions to logits add exactly and must be recombined before softmax:

```text
Q K^T = sum_g Q_g K_g^T.
```

Token blocks cannot be softmaxed independently and naively added. They can, however, be summarized exactly by the stable online-softmax state `(m,l,o)` and merged associatively. Thus split-KV is an algebraic factorization of the softmax/value computation, not merely a GPU scheduling trick.

GQA head space is different: query outputs are independent, but they share K/V operands. This makes the head axis a natural reuse/batched-matrix axis rather than an independent reconstruction axis.

---

## 12. Exact versus approximate dimension reduction

### 12.1 Instantaneous exact reduction

For a fixed Q, choose an orthonormal basis `U_Q in R^(d x rho_Q)` of the visible row space. Then

```text
Q K^T = (Q U_Q) (K U_Q)^T.
```

This reduces the exact feature contraction from `d` to `rho_Q`. It is not immediately useful for persistent caching because `U_Q` changes with Q and computing `K U_Q` online can cost as much as the original operation.

### 12.2 Windowed/effective subspace

Stack query groups over a decode window:

```text
Q_window = [Q_t; Q_{t-1}; ...].
```

The useful quantities are:

- exact rank;
- singular spectrum / effective rank;
- principal-angle drift of the dominant subspace;
- energy captured by hardware-aligned 32-wide blocks after a candidate basis transform.

If the dominant subspace is stable, the large instantaneous nullity may become exploitable. If it rotates rapidly to full dimension, exact persistent block deletion is not available.

### 12.3 Controlled block approximation

For a projector `P_k` onto selected feature blocks,

```text
E_score = Q(I-P_k) K^T
```

and

```text
||E_score||_F <= ||Q(I-P_k)||_F ||K||_2.
```

This creates a bound-driven approximation target rather than a heuristic sparsity threshold. The downstream softmax/output operator may reduce or amplify this error; the final operator error must be measured separately.

---

## 13. Error should be measured in the operator-visible space

Raw reconstruction metrics such as `||K-K_hat||_F` and `||V-V_hat||_F` treat null directions as important even when attention cannot observe them.

More relevant K error is

```text
E_K = Q (K-K_hat)^T C_L,
```

or, after softmax linearization,

```text
J_softmax(z) E_K.
```

For V, a more functional metric is

```text
E_V = P (V-V_hat)
```

and, if the following output projection is fixed,

```text
E_VO = P (V-V_hat) W_O.
```

This does not license arbitrary approximation. It identifies the norm in which an approximation should be judged.

---

## 14. Probability concentration gives a separate V-side approximation axis

For one query, `p_t >= 0` and `sum_t p_t=1`. If a dropped token set `D` has probability mass

```text
delta = sum_{t in D} p_t,
```

then

```text
|| sum_{t in D} p_t v_t ||
<= delta * max_t ||v_t||.
```

Useful diagnostics include

```text
N_eff = 1 / sum_t p_t^2
```

and the smallest support size carrying `1-delta` mass. This is a different structural axis from feature-space nullity and should not be mixed with it.

---

## 15. Circuit-aware mapping on H200/Hopper

The mathematical decomposition should be mapped to real primitives before any broad kernel sweep.

Relevant verified facts:

- NVIDIA PTX exposes `.u4/.s4` integer MMA shapes with reduction dimension `K=32` (for example `m8n8k32` and `m16n8k32` families; architecture requirements are satisfied by SM90/Hopper).
- CUTLASS documents Hopper mixed-input kernel generation for combinations including `int4 x f16/bf16`, with conversion/scale/zero-point modes.
- Therefore the current quantization group width `32` is aligned with an available sub-byte MMA reduction granularity, but this **does not prove** that the current Triton lowering uses that path or that a particular mixed-dtype path is one hardware instruction. Generated PTX/SASS must be inspected before making such a claim.

For the natural contraction

```text
(r x 32) @ (32 x token_tile),
r=6,
```

hardware tile utilization also matters. A nominal `K=32` match can still waste M/N lanes or require packing multiple logical objects together. Circuit-aware cost must therefore count at least:

```text
HBM bytes
packed-code bytes
scale bytes
number/type of MMA tiles
M/N/K tile utilization
conversion instructions
scale/epilogue instructions
register/shared-memory footprint
transform/RoPE overhead
```

The target is not “use Tensor Cores” in the abstract. It is to construct an operator factorization whose natural dimensions map efficiently onto available circuit primitives.

---

## 16. Q representation is the main remaining mixed-type question

K/V are stored in low-bit form, while Q and probability-weighted coefficients are currently floating point. Several possibilities must be distinguished:

1. **mixed-dtype hardware/software path**: use an available Hopper/CUTLASS mixed-input kernel if its generated lowering is efficient for the exact small-GQA geometry;
2. **Q block quantization**: represent each 32-wide Q block as a small integer vector plus a scale and move both scales outside the inner integer contraction;
3. **exact radix/digit decomposition**: because FP16 values have finite binary representations, a block can be expressed exactly using enough integer bit/digit planes plus powers-of-two scales; truncating planes gives a controlled approximation.

Hadamard/other mixing may reduce coordinate dynamic range and therefore reduce the number of planes needed for a chosen error tolerance. This is a testable link from basis geometry to actual MMA count, not merely to reconstruction MSE.

No choice is promoted yet. The first experiment must establish the geometry and the hardware-lowering opportunity.

---

## 17. Derived research questions

The next questions should be structural, not broad timing sweeps.

### Q1. Is instantaneous nullity exploitable over time?

Measure query-subspace rank/effective rank and principal-angle drift over decode steps/layers. Determine whether a stable low-dimensional visible subspace exists.

### Q2. Can visible energy be concentrated into a small number of 32-wide hardware blocks?

For candidate exact basis transforms, measure

```text
capture(k) = || Q P_k ||_F^2 / ||Q||_F^2
```

for `k=1..8` blocks, together with the operator-visible score error after dropping the tail.

### Q3. What does Hadamard actually buy in circuit terms?

Measure, before/after `H_32`, full `H_256`, and other structured bases:

- coherence / `||x||_inf / ||x||_2`;
- per-block exponent/dynamic-range spread;
- integer/radix plane count needed for exact representation or a fixed operator-error budget;
- group-scale matrix singular spectrum.

The primary output is not latency.

### Q4. How concentrated is V work in token space?

Measure `N_eff`, top-k probability mass, and final-output error bounds/actual errors for controlled support truncation.

### Q5. Does the resulting algebra map to fewer real circuit operations?

Only after Q1–Q4, implement the smallest representative microkernel and count/inspect:

- packed bytes loaded;
- scale operations;
- generated MMA/conversion instructions;
- PTX/SASS tile shapes;
- measured latency as a confirmation of the predicted work reduction.

---

## 18. Immediate experiment order

The current recommendation is:

```text
E10-E12: preserve as evidence about the old executor
E13: keep as optional/diagnostic phase-timing tool; not the scientific gate

E14A: query operator geometry
  - rank/effective rank
  - 32-block concentration
  - subspace drift

E14B: quantizer/basis geometry
  - Hadamard/coherence
  - scale-matrix spectrum
  - integer-plane requirement
  - operator-visible K error

E14C: V token/output geometry
  - N_eff / mass concentration
  - output-projected error

then

E15: circuit-mapped contraction prototype
  - representation-native 32-wide contraction
  - PTX/SASS verification
  - exact numerical gate first
  - timing only after work-count prediction is frozen
```

Do not redesign the production hot kernel before these structural measurements say which algebraic reduction is actually available in the model distribution.

---

## 19. Claims that are not yet established

Do not currently claim that:

- the E10 grouping crossover is a universal representation law;
- instantaneous `d-rho_Q` nullity permits persistent KV deletion;
- Hadamard reduces arithmetic dimension by itself;
- a 32-wide quantization group automatically maps efficiently to one Hopper Tensor Core path;
- CUTLASS mixed-dtype support implies one specific native mixed-dtype MMA instruction sequence;
- query block concentration, scale low-rank, or probability concentration is present before measuring it on the target model;
- a new contraction is faster until exact work counts, generated code, and timing all agree.

---

## 20. Prior-art anchors to keep in scope

These are not an exhaustive novelty search.

- **FlashAttention** (Dao et al., 2022): exact attention redesigned from IO complexity rather than naive materialization.
- **QuIP#** (Tseng et al., 2024): randomized Hadamard incoherence processing for low-bit quantization.
- **QuaRot** (Ashkboos et al., 2024): exact rotation invariances used to enable low-bit LLM inference including KV cache.
- **QServe** (Lin et al., 2024): W4A8KV4 serving, explicitly emphasizing dequantization/CUDA-core overhead and system/quantization co-design.
- **RotateKV** (Su et al., 2025): rotation placement relative to RoPE and low-bit KV quantization.
- **Provable Quantization with Randomized Hadamard Transform** (Feng et al., 2026): theoretical RHT quantization guarantees.
- **NVIDIA PTX ISA / CUTLASS Hopper documentation**: available sub-byte MMA and mixed-input kernel families; generated-code verification remains required for OFD's exact path.

The prospective novelty, if any, must come from the specific operator-quotient / representation-native / circuit-mapped construction and its demonstrated benefit, not from rotation, low-bit KV quantization, GQA reuse, or split softmax individually.
