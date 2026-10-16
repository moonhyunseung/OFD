# E11 controlled interventions — findings

Source: `3ceee31f696694b0e224b8f30085f59162682ec4`. 45 conditions, 3 independent processes, 3 shuffled rounds, 10 warmup and 50 timed calls/block. **2736/2736 timed blocks, 912/912 oracle checks PASS**, worst relative L2 0.0002112146467, finite outputs; no OOM. Exact per-process KV buffer identities were verified across capacity, alias and Q/KV interventions. Raw archive: `76e7bd8b3a9c42de56cafef9f3f5ecaffcf49a03`.

## 1. Direct observations

Ratios below are within-process/round, then median across nine blocks; lower means faster. All effects, full ranges and process-mean ranges are retained in CSV. No outlier removal or significance threshold.

| Intervention, g1 | INT4 latency ratio | FP16 latency ratio | Fixed axes / residual coupling |
|---|---:|---:|---|
| 32K/B4: capacity 1x -> 4x | 3.546942 | 3.145006 | Same actual Q/KV, length and active work; neutral programs, workspace and merge grow |
| 32K/B4: unique KV addresses 1x -> 1/16 | 0.931565 | 0.935754 | Same buffers, values, logical work and program count; block-table aliasing changes addresses |
| 32K: (B,Hkv)=(1,16) -> (16,1) | 0.952196 | 0.982942 | Same total heads/work/KV bytes; B, head geometry and strides change together |
| 32K/B4: Q/KV 1 -> 6 | 2.851216 | 3.039864 | Same KV buffers/bytes; Q geometry, head work and program count grow |

## 2. Empty-split overhead is a major finding

At actual 32K/B4, cap=128K has four times as many launched splits while actual tokens, Q and KV buffers are identical. INT4 g1 becomes 3.54694x slower (three process-mean ratios 3.54593..3.54716); FP16 3.14501x (3.13983..3.14626). Thus active KV traversal/bytes alone cannot explain this executor latency. This is also evidence against treating every launched program as equivalent to useful token work.

Source inspection: `subgrouped_kernels.py` traverses a fixed 64 tile iterations per split (`1024/16`), masks loads for invalid tokens, and writes neutral partials at the end. It does not skip the whole loop for a fully inactive split. This is a source-level observation, not a profiler measurement of dynamic instructions. Capacity also grows partial workspace and Stage-2 merge; E11 does not isolate those contributions. No kernel was changed to remove the cost.

Grouping D(g2) at 32K/B4 rises from 1.02946ms at cap1 to 3.83308ms at cap4 with KV unchanged. GI rises 0.50190->1.77202ms and GF becomes -0.52774->-2.06074ms. Therefore D and grouping gain cannot be attributed exclusively to reduced KV-memory traffic. Repeated packed arithmetic on masked tiles, compiler resources, scheduling and merge remain possible contributors; E11 does not establish their percentages.

## 3. Unique-address footprint matters, but is not a sufficient explanation

At 32K/B4/g1, reducing unique referenced pages sixteen-fold changes INT4 latency to 0.931565x and FP16 to 0.935754x. Allocated KV bytes are unchanged (INT4 144MiB and FP16 512MiB); unique referenced bytes become 9MiB and 32MiB. This is address reuse, not a compressed allocation or a serving capacity gain.

The effect is not uniformly beneficial: FP16 8K/B1/g1 ratio is 1.0503 and FP16 32K/B4/g3 is 1.0609. All settings remain in contrasts_summary.csv. A sixteen-fold unique-address reduction does not create a sixteen-fold latency reduction. These observations refute a simple footprint-proportional model in this synthetic test; they do not prove HBM bandwidth is irrelevant or measure actual HBM traffic.

## 4. Batch-axis repartition

At fixed T=32K, B*Hkv=16 and Q/KV=6, total query heads=96, KV elements, active program count and merge output count match. Going B1/Hkv16 -> B16/Hkv1 gives g1 ratios INT4 0.952196 and FP16 0.982942. There is no adverse monotonic high-batch collapse in this g1 control. Observed timings are not numerically invariant under this factorization; the stable process ratios support an effect within this matrix. Head shape and memory strides change with B, so this is not pure batch causality. Aggregate tokens/s across these different head geometries is not a fair model throughput comparison; latency and total query-head throughput are the appropriate diagnostics.

## 5. Q-head repetition with fixed KV

At 32K/B4/g1, Q/KV 1->6 grows INT4 latency 2.85122x and FP16 3.03986x while the same KV buffers/bytes remain. More query work therefore changes latency without any increase in KV allocation. The relative response at this shape is similar, not an INT4-only phenomenon. At low work, INT4 8K/B1 R6/R1 is 0.9024, showing that more query heads can improve parallelism enough to lower latency. The dependence is not universally linear or monotonic.

At 32K/B4, INT4 g2 saves about 0.60640ms for R2 and 0.50132ms for R6; D is about 0.90597ms and 1.02893ms. More repetition does not automatically produce proportionally greater grouping gain. These interventions change arithmetic/program count and query geometry together; they do not isolate integer-unpack instruction cost.

## 6. Secondary cross-family triangulation

`secondary_equal_program_footprint_raw.csv` is a secondary diagnostic, not the frozen primary statistic. A=actual8K/cap32K/no alias; B=actual32K/cap32K/alias4; C=actual32K/cap32K/no alias. A and B have equal launched programs, equal unique referenced KV bytes and equal workspace dimensions, while B has four times the active token traversal. A/B use different allocations across T, so identical physical pointers are not claimed. B/C share exact buffers.

At B4/g1, median B/A is 1.07017 for INT4 and 1.18868 for FP16; median C/B is 1.01128 and 1.01316. Masked and useful traversal are not free/equivalent, but the large empty-split overhead remains even without additional active KV reads. Other g values and all rounds are retained; no universal bottleneck is inferred.

## 7. Supported, rejected and unresolved interpretations

**Supported:** latency depends on representation, launched masked work, execution granularity, query repetition and physical address reuse in different ways. Neither KV bytes alone nor useful token count alone determines latency. Large grouping gain can persist/grow without new active KV loads. A single representation-independent count is insufficient across all interventions.

**Rejected within this matrix:** footprint-proportional latency; every launched split corresponding to useful traversal; systematic high-batch g1 collapse at fixed aggregate heads/work; universal linear query repetition cost; D being a direct measurement of KV traffic removed.

**Unresolved:** dynamic instruction attribution, register pressure/spills, HBM vs cache traffic, Stage-1 vs merge timing decomposition, and generality beyond this kernel/GPU/config. Representation-dependent scheduling is still a live explanation; E11 does not make every dependency claim impossible. Testing variables is intended to expose actual dependencies rather than force independence.

## 8. Scope and next candidate

Pages intentionally contain identical values across logical positions/heads so aliasing and repartition preserve attention outputs exactly. This is a synthetic causal-control suite, not realistic traffic or production serving. Seeds differ across the three processes. Existing E00-E10 protocols/results and main are unchanged. No nibble, quantizer, precision, split/tile/warp/stage, cache-format or merge optimization was introduced.

The next evidence-backed candidate is skipping completely inactive Stage-1 split loops, in a separately committed optimization experiment with correctness and matched FP16 controls. Stage-1/merge decomposition and profiling would then attribute the remaining cost. No speedup from that unimplemented change is claimed.

## Provenance and failures

The initial import attempt failed before measurement because PYTHONPATH was absent; setup_import_failure.log is retained. Initial source 57ac324 ran a smoke and an interrupted preliminary full sweep before exact-buffer reuse was strengthened. Those directories are retained and excluded from claims. The valid smoke and full sweep use source 3ceee31 only. See MANIFEST.json, hardware.json, analysis.json and ARCHIVE_HASHES.json.
