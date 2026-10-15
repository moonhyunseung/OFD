# E10: Qwen-geometry execution grouping

The completed standalone causal sweep is in `20261015T161635Z_72911/`. See its `FINDINGS.md`, `VALIDATION.json`, raw CSV and MANIFEST. Earlier smoke directories preserve the rejected rank-3 g1 timing comparison and the repaired scalar-g1 check.

## Three separate answers

**Q1 — total work:** Equal W1 is a strong organizing axis. In the requested equal-work triplets, INT4 g1 differs by at most about 3.7% and FP16 by about 0.5% relative to the first member. This does not prove program count alone is causal: equal W1 also fixes B*T and KV footprint, and representation changes per-program work/resources.

**Q2 — independent batch penalty:** The requested triplets do not show larger batch systematically worsening latency at equal W1. The slight INT4 ordering is in the opposite direction. This rejects that ordering in these measurements, not every possible batch effect; context changes together with batch.

**Q3 — INT4-specific grouping benefit:** High-work INT4 improves while matched FP16 grouping slows down. At 128K/B4/g2, paired GI=1.81543ms, GF=-2.09344ms, D=3.90864ms. Positive D includes the FP16 penalty; it is not 3.91ms of INT4 work removed. Low-work INT4 can slow down even with positive D. The intervention supports representation-dependent benefit consistent with packed KV reuse, but does not isolate unpack cost from representation-dependent scheduling/compiler resources.

## Validation and frozen selection

24 shapes, 3 independent processes, 3 shuffled rounds per process, 10 warmup and 50 timed calls per configuration. 2160 timing blocks include 432 legacy controls. All 648 numerical comparisons pass, worst relative L2 1.65262e-5; no OOM. An isolated FP16 g1 timing excursion is retained and documented; repeated g1 implementation drift is absent after repair.

Standalone capacity is exactly T and Ns=ceil(T/1024). All requested geometry and execution knobs remain frozen. Legacy Stage-2 merge, quantizer, nibble decode and storage layout are retained. Non-power-of-two logical groups are masked within padded query axes.

`g*_FP16=1` in all 24 shapes. INT4 shifts toward larger g with work; equal-W1 shapes share the same observed optimum. The frozen full-model policy chooses **one fixed g2 for INT4 and g1 for FP16**, using the all-shape/all-block geometric mean; it was committed before model execution. No posthoc per-shape choice is used. Even the best measured INT4 g remains slower than FP16 g1 at every standalone shape.

## Claim limits and next attribution

Directly measured: latency, throughput, finite/equivalent outputs, grouping effects, difference-in-differences, equal-work contrasts and g* surface. Supported interpretation: high-work packed execution benefits from reduced q-head repetition, with a representation-dependent response.

Unverified: exact HBM bottleneck, spill behavior, or integer-unpack instruction percentage. Those require profiler attribution after this causal test. No split/warp/nibble tuning, new quantizer, rotation/layer search, model or dataset was introduced. The next useful work is profiling the frozen paths to distinguish packed reuse from compiler/resource scheduling effects before selecting an optimization.

## Provenance

- Implementation: 5cdaeb157256f0be4a15b828730668a1f8c42b55
- Repaired low-level source: 6ef2a73ec39f33743cc36ccc49bb93b2c373b8e8
- Raw low-level archive: 78ce8b61121ad132719589fde92e13305e5642c6
- Analysis, validity gate and frozen policy: 17d298f54de8d04a61d825f5ed2868e1cd4b37ff
- Model confirmation source: 27092f57dfa2e06d4aa56a0977bc431b52bc9af3

Existing E00–E09 protocol/result history is preserved. The optional OnlineKV argument defaults to g1 and preserves its original execution. Main is unchanged by E10.

## Complete-model confirmation

`confirmation_20261015T163740Z_7bd99095/`: 96/96 rows complete, no OOM/skips, source 27092f57dfa2e06d4aa56a0977bc431b52bc9af3. Smoke has 12 model rows plus 16 online append equivalence checks. Frozen policy is unchanged. Raw confirmation archive: 887ccf5b6d601c076899ee1aab60625e144743c7.

| T | B | INT4 g1 ms | INT4 fixed g2 ms | Paired speedup median | Paired speedup range | FP16 g1 ms |
|---|---:|---:|---:|---:|---|---:|
| 8192 | 1 | 54.776 | 54.889 | 0.9990 | 0.9933 .. 1.0029 | 53.742 |
| 8192 | 8 | 59.838 | 56.705 | 1.0552 | 1.0526 .. 1.0561 | 53.991 |
| 32768 | 1 | 55.031 | 54.662 | 1.0073 | 1.0067 .. 1.7503 | 54.131 |
| 32768 | 4 | 71.603 | 63.461 | 1.1281 | 0.7170 .. 1.5951 | 54.885 |
| 65536 | 2 | 69.659 | 61.218 | 1.1380 | 1.1376 .. 1.1517 | 55.148 |
| 65536 | 4 | 109.684 | 93.756 | 1.1699 | 1.1698 .. 1.1699 | 54.294 |
| 131072 | 1 | 69.161 | 60.394 | 1.1442 | 0.8216 .. 1.1452 | 53.651 |
| 131072 | 4 | 184.237 | 154.622 | 1.1916 | 1.1910 .. 1.1916 | 61.671 |

Direct model observations: g2 reduces INT4 median latency materially at larger workloads. The most repeatable paired improvements in this process are 8K/B8 (1.0526..1.0561x), 64K/B2 (1.1376..1.1517x), 64K/B4 (1.1698..1.1699x), and 128K/B4 (1.1910..1.1916x). At 128K/B4, INT4 median latency falls 184.237->154.622ms (about 16.1% latency reduction, 19.2% throughput increase). At 8K/B1 there is no improvement. 32K/B1's small median gain and the broad ranges at 32K/B4 and 128K/B1 are not robust evidence by themselves; those ranges include reversals or large excursions.

The identical FP16 g1/gbest control has large excursions in several shapes (for example 8K/B1 paired ratio up to 1.8546). No outlier is removed. Three rounds share one model process, so model confirmation does not have the independent-process replication of the standalone causal sweep. The consistent high-work cases support transfer to this complete-model decode stack, while the noisy cases remain uncertain.

Every fixed-g INT4 shape remains slower than FP16 g1 by median latency; at 128K/B4, 154.622ms vs 61.671ms (2.507x latency). KV bytes remain exactly 3.5555556x smaller at the same batch/context. This confirmation changes grouping, not capacity; it does not establish a same-memory-budget serving throughput win.

All 64 layers, online append, LM head and teacher-forced decode remain in scope; prefill/setup/sampling and serving scheduler are excluded. Native GDN uses the existing PyTorch reference fallback because optional optimized dependencies are absent. No native logits accuracy reference is collected in this speed-only confirmation. See each MANIFEST, data_manifest and CONFIRMATION.md. Production serving and optimized-native-stack performance are not established.

System interpretation: high-work core gains can reach whole-model latency without eliminating the remaining INT4/FP16 gap. The causal attribution remains representation-dependent execution/reuse; exact HBM traffic, spill, and unpack instruction contributions are still hypotheses. The next candidate is profiler attribution on frozen g1/g2/g6 and FP16 controls, followed by a separately versioned optimization experiment. No new optimization was mixed into E10.
