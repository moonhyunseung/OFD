# Architecture

## Separation of concerns

```text
reference/    = correctness oracle
triton/       = hardware execution drafts
runtime/      = launch/workspace/policy interface
experiments/  = frozen scientific protocols
results/      = generated evidence, never the source of truth by itself
docs/         = mathematical/operator contracts and claim boundaries
```

## Architectural rule

OFD should no longer be designed by starting from a dequant-first Triton loop and tuning it empirically.

The design order is:

```text
operator geometry
  -> exact/approximate factorization
  -> placement across nonlinear boundaries
  -> hardware-native contraction shape
  -> implementation
  -> measurement
```

See `OPERATOR_GEOMETRY_AND_CIRCUIT.md` for the current design model.

## Current executor (E00-E12 lineage)

The frozen correctness-first executor is:

```text
packed K page
  -> nibble decode / scale to working values
  -> QK contraction
  -> online softmax state

packed V page
  -> nibble decode / scale to working values
  -> weighted V accumulation
  -> split partial (m,l,o)
  -> merge
  -> optional output transform
```

Dense **cache** materialization is forbidden, but the inner loop still conceptually reconstructs working K/V values before contraction. E10-E12 characterize this executor and remain valid historical evidence.

The E10 GQA-grouping crossover must not be assumed to be a fundamental law of compressed attention: it may partly reflect the dequant-first contraction order of this executor.

## Candidate next architecture

The candidate operator-native path is qualitatively different:

```text
Q / RoPE / exact basis placement
  -> hardware-aligned 32-wide query blocks

packed K codes + group scales
  -> representation-native 32-wide contraction
  -> scale/representation correction after reduction
  -> sum feature-group score contributions
  -> logits

logits
  -> online softmax sufficient statistics

probability coefficients + V scales
  -> representation-native packed-V contraction
  -> output in chosen V basis
  -> inverse basis absorbed into following W_O when exact/legal
```

No dense dequantized K/V object is part of the target operator graph.

## Operator placement rules

Before adding an operation to the hot path, classify it:

1. **commuting operator**: move to the cheaper side if `AB=BA`;
2. **cheaply conjugatable operator**: move across `B` if `A^{-1}BA` remains structured;
3. **adjoint-transferable operator**: move across a bilinear pairing using `A*`;
4. **direct-sum distributive operator**: split along independent feature/token blocks;
5. **nonlinear boundary**: stop unless a sufficient-statistic/factorization theorem applies.

Quantization is a nonlinear boundary. Softmax is nonlinear but admits exact token-block reduction through `(m,l,o)`.

## Basis policy is not symmetric across Q/K/V

Do not assume one transform belongs at the same location on Q, K, and V.

- **Q/K side**: placement is constrained by the score bilinear form, RoPE conjugation, and quantizer geometry.
- **K side**: only the query-visible quotient matters for the current score operator, but its nullspace is dynamic.
- **V side**: no RoPE barrier; an inverse feature transform can potentially be absorbed into the following output projection exactly.
- **GQA head axis**: shared K/V makes head grouping a reuse/matrix-shape issue, not a reason to reconstruct K/V per head.

## Hardware mapping rule

The mathematical block decomposition must be mapped to a real circuit primitive before broad benchmarking.

Current group width is 32. Hopper/PTX exposes signed/unsigned 4-bit MMA families with reduction dimension 32, and CUTLASS documents Hopper mixed-input kernels involving int4 and f16/bf16. This makes 32 a plausible circuit-aligned block width, not an automatic speed guarantee.

A candidate implementation is not accepted because it “uses Tensor Cores.” It must show:

```text
exact logical contraction
-> exact storage-code interpretation
-> concrete MMA/conversion/scale lowering
-> tile utilization for r=6 and token tile
-> predicted byte/instruction reduction
-> generated PTX/SASS confirmation
-> numerical gate
-> timing confirmation
```

## Current experiment priority

E13 phase timing is retained as a diagnostic tool but is no longer the main scientific gate.

The next primary work is operator-structure probing:

- query rank/effective rank and subspace drift;
- concentration into 32-wide hardware blocks under exact basis choices;
- Hadamard/coherence/dynamic-range and integer-plane consequences;
- scale-matrix spectrum;
- V probability/support concentration and output-visible error.

Only after these measurements identify an exploitable structure should a new hot kernel lineage begin.
