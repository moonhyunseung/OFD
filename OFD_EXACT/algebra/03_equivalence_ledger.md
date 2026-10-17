# 03 — Equivalence ledger

This ledger is the gate between algebra and implementation.

No candidate enters implementation unless its exact status is settled here.

| ID | Original computation | Candidate form | Status | Guarantee / theorem | Required assumption | Reconstruction / transformed next map | Algebraic benefit |
|---|---|---|---|---|---|---|---|
| K1 | decode `(C_K-8)S_K`, then `Q Khat^T` | packed code contraction + post-reduction scale + scalar zero-point correction | **EXACT** | distributivity / associativity | frozen decode contract | none | removes dense K dequantized intermediate; fewer scale / zero-point coordinate ops |
| V1 | decode `(C_V-8)S_V`, then `P Vhat` | coefficient prescale `P*S_V`, code contraction, broadcast zero-point correction | **EXACT** | distributivity / associativity | frozen decode contract | none | removes dense V dequantized intermediate |
| P1 | two nibbles -> dense code vector | even/odd packed-pair contractions and sum | **EXACT** | direct coordinate partition | nibble ordering known | none | avoids logically duplicated packed-byte consumption if lowering supports reuse |
| B1 | uint4 code contraction | four exact bit-plane contractions and weighted sum | **EXACT** | radix decomposition | unsigned nibble contract | none | exposes alternative binary primitive mapping; no guaranteed arithmetic win |
| H1 | `Q,K` | `QH,KH` for orthogonal H | **EXACT ISOMORPHISM** | `H^T H=I` | H orthogonal | inverse / dual action if later state expects original basis | changes coordinate geometry only; cannot reduce rank or singular values |
| H2 | block H32 proposed to concentrate group energy | `I_8 ⊗ H_32` | **REJECT AS CONCENTRATOR** | orthogonal norm preservation inside each existing block | none | n/a | block energy is invariant, so this goal is impossible |
| S1 | arbitrary exact SVD/rank factorization of Q | `Q=AB` with inner dim `rank(Q)` | **EXACT** | rank factorization theorem | none | none | no exact inner dimension below rank(Q) |
| S2 | truncated SVD of Q/P | rank-k with `k<rank` | **APPROXIMATE** | not equal in general | downstream-invisible truncated subspace would be required | otherwise reconstruct impossible | excluded from exact path |
| N1 | persistent K feature deletion | quotient by `N=∩ ker(Q_t)` | **EXACT CONDITIONAL** | first isomorphism theorem / common kernel | scope of future Q family fixed; `N` nontrivial | none if every future score factors through quotient | exact feature-state reduction only if common kernel exists |
| SM1 | logit constant direction | quotient by `span{1}` | **EXACT QUOTIENT** | softmax translation invariance | row-wise softmax | none | removes one token-axis logit degree of freedom semantically |
| TK1 | split token attention | online-softmax sufficient statistics and associative merge | **EXACT** | stable softmax merge identity | correct block state `(m,l,o)` | merge | enables token blocking without approximate softmax |
| R1 | move basis A across RoPE | conjugated RoPE `A^-1 R_p A` | **EXACT CONDITIONAL** | conjugacy | A invertible | transformed RoPE mandatory unless A commutes | free movement only in commutant / cheap normalizer |
| G1 | move dense V basis A across dynamic diagonal gate D_t | `A D_t A^-1` | **EXACT BUT STRUCTURALLY CONDITIONAL** | conjugacy | A invertible | transformed dynamic gate mandatory | generally unattractive if conjugation destroys diagonal structure |
| SR1 | scale matrix S | `S=a b^T` | **EXACT CONDITIONAL** | rank-1 factorization iff `rank(S)=1` | exact rank one | none | separates token and group scale dependence |
| SR2 | scale matrix S | numerical rank-1/2 approximation | **APPROXIMATE** | Eckart–Young only gives error-optimal approximation | error budget | explicit approximation branch | excluded from exact path |

## Status meanings

- `EXACT`: pointwise equality under the stated frozen semantics.
- `EXACT ISOMORPHISM`: information-preserving representation change; downstream map must respect the new coordinates.
- `EXACT QUOTIENT`: non-injective map whose discarded distinctions are provably invisible to the scoped downstream observable.
- `EXACT CONDITIONAL`: theorem is exact if an explicitly testable structural assumption holds.
- `EXACT BUT STRUCTURALLY CONDITIONAL`: equality can be retained only by transforming a neighboring operator, which may destroy useful structure.
- `APPROXIMATE`: not part of the exact path.
- `REJECT`: the stated intended benefit is ruled out algebraically.

## Required fields for every new row

Every future candidate must state:

1. source and target sets/spaces;
2. exact formula;
3. theorem/identity providing the guarantee;
4. all assumptions;
5. inverse, reconstruction, quotient, or conjugated downstream map when applicable;
6. symbolic operation/intermediate-state change;
7. the precise boundary that remains empirical.