# Claims and gates

Claim eligibility requires a clean Git worktree unless a protocol explicitly declares otherwise. Historical claims remain tied to their original experiment lineage.

| Claim | Minimum gate |
|---|---|
| CPU/storage/reference correctness | E00 PASS |
| Triton implementation correctness | E02 PASS |
| Core packed-INT4 attention has <=5% OFD-vs-Plain overhead on the frozen matrix | E04a PASS |
| GPU query-transform + attention + output-transform has <=5% overhead vs Plain INT4 on the frozen matrix | E04b measured worst ratio <=1.05 after transform correctness; not production serving E2E |
| Compression improves same-KV-budget throughput in a stated region | E05 measured per-case throughput ratio >1 with matched paged FP16 baseline; residency alone is insufficient |
| Grouped GQA is faster in a stated region | historical grouped experiment correctness + measured speedup in that region; not a universal law |
| Long-context curve is measured | E07 COMPLETED (`gate_status=NOT_APPLICABLE`) |
| Current executor has representation-dependent grouping preference | E10 matched INT4/FP16 intervention, limited to tested geometry/workload |
| Launched masked work materially affects padded-capacity latency | E11 capacity control + E12 inactive-loop intervention |
| E12 guard is production-serving beneficial | **NOT ESTABLISHED**; measured complete-model path had no wholly inactive split at the tested starting points |
| E10 grouping crossover is a fundamental compressed-attention law | **NOT ESTABLISHED**; must survive a representation-native contraction redesign |
| Instantaneous query nullity can remove persistent KV dimensions | E14A must show stable/effective subspace across declared windows/layers/heads; one-step rank is insufficient |
| A basis concentrates query-visible work into fewer 32-wide blocks | E14A representative sampling + frozen basis/search space + captured-energy and operator-visible score-error tables |
| Hadamard reduces circuit work | E14B must show reduced conversion/plane/tile requirement at fixed operator-error budget; lower coherence/MSE alone is insufficient |
| K approximation preserves the attention-relevant operator | report `Q Delta K^T C_L` and downstream effect under declared norm/tolerance across representative samples |
| V approximation/support reduction preserves output | E14C dropped-mass bound + measured `P Delta V` and, where available, `P Delta V W_O` across representative samples |
| A 32-wide low-bit contraction maps to Hopper efficiently | E14D/E15 static tile ledger + generated PTX/SASS + numerical gate; source-level dtype/shape is insufficient |
| New operator-native microkernel reduces hardware work | E15 predicted byte/op/MMA/tile ledger frozen before run + generated-code confirmation + measured work agreement |
| New operator-native microkernel is faster | E15 numerical gate + generated-code verification + timed comparison on frozen conditions |
| New operator-native design survives real-model decode | complete-model integration after E15 microkernel gate, with fixed policy and no favorable protocol changes |
| Production serving / same-memory-budget advantage | matched optimized serving baseline and equal-budget throughput/capacity experiment; microkernel or teacher-forced decode alone is insufficient |

## Exactness classes

Every new transformation must be labeled before implementation:

- `EXACT`: identical output for all admissible inputs;
- `CONDITIONAL_EXACT`: identical under a declared structural condition/subspace;
- `BOUNDED_APPROX`: deterministic operator-level bound declared;
- `PROBABILISTIC_APPROX`: explicit distribution/concentration guarantee declared;
- `HEURISTIC`: exploratory only; not a mechanistic claim.

Do not silently promote a heuristic into an exact/causal statement because a benchmark improved.

## Operator-visible error rule

Raw reconstruction error is not enough when the operator has a large kernel/null space.

At minimum:

```text
K: report Q (K-K_hat)^T C_L
V: report P (V-V_hat)
```

and, when the following output projection is fixed and available:

```text
V/output: report P (V-V_hat) W_O.
```

The declared error norm/tolerance must be frozen before claim-eligible analysis.

## Generated-code rule

Instruction-level and Tensor-Core claims require generated PTX/SASS or an equally direct compiler artifact. Library support or high-level source types are not sufficient evidence of the emitted path.

## Historical integrity

E00-E13 storage encoding, quantizer, basis policy and entry-point semantics remain frozen for reproduction. A signed-s4 recoding, new basis, new quantizer, new contraction order or circuit-native microkernel belongs to a new lineage.
