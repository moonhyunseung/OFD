# Frozen predecessor contract

## Immutable predecessor

The historical OFD measurement line is frozen at:

```text
branch: ofd-v2.3-measurement-safe
commit: 58bbec23ba1c862f98b71af014c4a56e9a9e7a88
```

The frozen line contains the accumulated E00-E14 work, including E10-E12 measurements, E13 diagnostics, and E14 exploratory structural capture/analysis.

## What the freeze means

1. Existing E00-E14 files are not modified by `OFD_EXACT`.
2. Existing measurements remain valid only for the implementation and protocol under which they were collected.
3. No exploratory interpretation from E14 is promoted automatically to an algebraic claim.
4. New derivations restart from the frozen representation and model contracts.
5. If later implementation work needs to compare against E00-E14, it does so by explicit SHA/protocol reference rather than by rewriting predecessor history.

## Provenance split

- **Historical implementation/measurement evidence:** predecessor branch and its commits.
- **Exact algebraic reconstruction and subsequent work:** `OFD_EXACT` on the new research line.

## Frozen quantization representation used as the first reference

The predecessor reference quantizer stores each group using:

- group width `m = 32`;
- one FP16 scale per group;
- 4-bit code `C = q + 8` packed low nibble first;
- decoded integer `q = C - 8`;
- decoded value `x_hat = scale * q`.

The encoder that produced those codes is lossy. `OFD_EXACT` does **not** claim equivalence to the original pre-quantized FP tensor. Exact executor claims begin after the packed code/scale state has been fixed.