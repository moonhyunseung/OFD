# 00 — Spaces and maps

This file defines the objects before any optimization is discussed.

## 1. Core spaces

For one GQA KV head, let

- query-head space: `H = R^r`, with current `r = 6`;
- token space: `T = R^L`;
- feature space: `F = R^d`, with current `d = 256`.

The quantizer partitions the feature coordinate into

```text
d = G * m = 8 * 32.
```

It is therefore useful to expose the tensor structure

```text
F ≅ Gspace ⊗ W,
Gspace = R^8,
W = R^32.
```

No dimensional reduction is implied by this identification; it is only a factorization of coordinates.

## 2. Linear maps

Using row-vector matrix notation:

```text
Q ∈ R^(r×d)
K ∈ R^(L×d)
V ∈ R^(L×d)
```

Interpretation as maps:

```text
Q : F -> H
K : F -> T
V : F -> T
K* : T -> F
```

The score bilinear map is

```text
B(Q,K) = Q K^T ∈ R^(r×L).
```

With scalar attention scale `α`, logits are

```text
Z = α Q K^T.
```

## 3. Nonlinear and downstream maps

Row-wise softmax is

```text
S : R^(r×L) -> (Δ^(L-1))^r.
```

Write

```text
P = S(Z).
```

Value aggregation is bilinear once `P` is fixed:

```text
C(P,V) = P V ∈ R^(r×d).
```

For the captured Qwen path, downstream computation may additionally contain a token-dependent diagonal gate `D_t` followed by output projection `W_O`.

Thus the semantic chain after Q/K/V formation is schematically

```text
(Q,K,V)
  -> Z = α Q K^T
  -> P = softmax(Z)
  -> Y = P V
  -> D_t Y
  -> W_O(...).
```

## 4. Representation spaces

The packed INT4 representation is not a vector space over `R`. For one K/V tensor define a representation set

```text
R4 = Codes × Scales,
Codes ⊂ {0,...,15}^(L×G×m),
Scales ⊂ R_+^(L×G).
```

The frozen decoder is a map

```text
D4 : R4 -> R^(L×d)
```

with coordinate formula

```text
D4(C,S)_[t,g,j] = S_[t,g] (C_[t,g,j] - 8).
```

The encoder is lossy in general. Therefore `R4` is treated as the semantic source for exact executor rewrites.

## 5. Classification of maps

Every edge in the chain must be labeled as one of:

- linear;
- bilinear;
- invertible linear / isomorphism;
- quotient projection;
- nonlinear but exact;
- lossy encoder;
- discrete representation map.

No optimization proposal is evaluated before its source/codomain and map class are explicit.