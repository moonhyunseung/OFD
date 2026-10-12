# Architecture

## Separation of concerns

```text
reference/  = correctness oracle
triton/     = hardware execution draft
runtime/    = launch/workspace/policy interface
experiments/= frozen scientific protocols
results/    = generated artifacts, not source of truth by itself
```

The hot attention kernel is basis-agnostic. K-rotation is handled by preparing q in the stored K basis. V-rotation is handled by transforming only the merged output. Runtime representation policy must not introduce token-level dynamic branching; the intended policy is static per layer.

## Target dataflow

```text
packed K page -> direct nibble unpack/dequant -> QK
                                            -> online softmax state
packed V page -> direct nibble unpack/dequant -> PV accumulation
                                            -> split partial (m,l,o)
                                            -> merge
                                            -> optional output FWHT
```

Dense dequantized KV materialization is forbidden in the target Triton path.
