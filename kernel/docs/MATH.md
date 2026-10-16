# Mathematical contract

This file is the compact mathematical contract. The full derivation and hardware-facing interpretation are in `OPERATOR_GEOMETRY_AND_CIRCUIT.md`.

## 1. Spaces and attention operator

For one KV head and the `r` query heads that share it:

- feature space `F = R^d`
- token space `T = R^L`
- query-head space `H = R^r`

Represent

```text
Q in Hom(F,H),
K,V in Hom(F,T).
```

The score operator is

```text
B(Q,K) = Q K^T,
```

row-wise softmax gives `P`, and value aggregation is

```text
C(P,V) = P V.
```

Thus

```text
A(Q,K,V) = softmax(QK^T) V.
```

## 2. Exact feature quotient for K

For fixed `Q`, keys are equivalent modulo `ker(Q)`:

```text
k ~ k+n,  n in ker(Q).
```

Hence the score depends on `F/ker(Q)`, whose dimension is `rank(Q)`. For the current Qwen GQA group, `Q` has shape `6 x 256`, so one-step rank is at most 6 and instantaneous nullity is at least 250.

This is an operator statement, not a persistent-storage claim: `ker(Q)` changes with the query.

## 3. Softmax quotient

For each query row,

```text
softmax(z + c 1) = softmax(z).
```

With `C_L = I - (1/L)11^T`,

```text
softmax(QK^T) = softmax(QK^T C_L).
```

For fixed `Q` with rank `rho_Q`, the pre-softmax observable map

```text
K -> QK^T C_L
```

has rank `rho_Q(L-1)` and nullity

```text
Ld - rho_Q(L-1).
```

## 4. V quotient

For fixed `P`,

```text
V -> PV
```

has rank `rank(P)d` and nullity

```text
(L-rank(P))d.
```

Functional V error should therefore be judged at least in `||P Delta V||`, or in `||P Delta V W_O||` when the following output projection is fixed.

## 5. Exact basis symmetry

For any invertible `A`,

```text
Q_A = Q A,
K_A = K A^{-T}
```

satisfy

```text
Q_A K_A^T = QK^T.
```

Orthogonal rotation is the special case `A^{-T}=A`.

For values, if `V_B=VB`, then

```text
(P V_B) B^{-1} = PV.
```

A V-side inverse transform can be absorbed into the following output projection when model layout permits:

```text
(P V B)(B^{-1}W_O) = P V W_O.
```

## 6. Hadamard

For normalized Hadamard `H`, `HH^T=I`, so

```text
(QH)(KH)^T = QK^T,
(PVH)H^T = PV.
```

Hadamard preserves rank/nullity; it does not reduce dimension by itself. Its quantization value comes from changing coordinate coherence/dynamic range while preserving the exact full-precision operator.

For `d=256=8*32`, the Sylvester structure can be written, up to ordering/sign convention, as

```text
H_256 = H_8 tensor H_32.
```

This lets group-internal mixing and cross-group mixing be reasoned about separately.

## 7. RoPE placement

For a position operator `Omega_t` and a basis transform `A`,

```text
Omega_t A = A (A^{-1} Omega_t A).
```

Commutation is sufficient but not necessary. A transform can be moved across RoPE when the conjugated operator remains acceptably structured/cheap. Generic basis motion is therefore a conjugation problem, not merely an associativity trick.

## 8. Quantization breaks the basis symmetry

Let `E,D` be the quantizer encoder/decoder. In general,

```text
D(E(KA)) != D(E(K)) A.
```

Thus the basis should be chosen before quantization to make the nonlinear representation favorable.

The target OFD contraction is representation-native:

```text
B_tilde(Q,c_K) = B(Q,D(c_K)),
C_tilde(P,c_V) = C(P,D(c_V)),
```

without materializing dense `D(c_K)` or `D(c_V)`.

## 9. Exact groupwise INT4 algebra

Current storage uses quantized values `q in [-7,7]`, encoded as unsigned nibbles `U=q+8`, group width `m=32`, and one FP16 scale per group/token.

For K group `g`:

```text
K_g = D(s_g^K)(U_g^K - 8 1 1^T)
```

and

```text
Q_g K_g^T
= (Q_g U_g^{K,T}) D(s_g^K)
  - 8 (Q_g 1) (s_g^K)^T.
```

Therefore scale application can move after the width-32 contraction, and the unsigned zero-point correction is rank 1.

For V:

```text
P V_g
= (P D(s_g^V)) U_g^V
  - 8 (P s_g^V) 1^T.
```

Dense dequantized K/V blocks are not algebraically required.

A future signed-s4 storage recoding could remove the `+8/-8` representation correction exactly, but that would be a new storage lineage.

## 10. Exact packed decompositions

For low/high nibbles `L_g,H_g` and even/odd Q coordinates,

```text
Q_g U_g^T = Q_g^even L_g^T + Q_g^odd H_g^T.
```

For bit planes,

```text
U = B0 + 2B1 + 4B2 + 8B3
```

so

```text
QU^T = QB0^T + 2QB1^T + 4QB2^T + 8QB3^T.
```

Both are exact; usefulness depends on hardware mapping.

## 11. Token splitting / stable softmax merge

For split `s`, store `(m_s,l_s,o_s)` with

- `m_s = max z_i`
- `l_s = sum exp(z_i-m_s)`
- `o_s = sum exp(z_i-m_s) v_i`

Global merge uses `m=max_s m_s`, weights `a_s=exp(m_s-m)`, denominator `sum a_s l_s`, and numerator `sum a_s o_s`.

The merge is associative in this sufficient-statistic representation. Empty splits are neutral: `m=-inf, l=0, o=0`, with safe masking before exponentiation.

## 12. Approximation must be operator-visible

For a feature projector `P_k`, score-tail error is

```text
E = Q(I-P_k)K^T
```

with

```text
||E||_F <= ||Q(I-P_k)||_F ||K||_2.
```

Raw reconstruction error is secondary to operator-visible error. Approximate designs must report both their mathematical bound/target norm and the actual downstream error.
