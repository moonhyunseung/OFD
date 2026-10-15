# H200 E03/E05 final report — 2026-10-15 (KST)

## Measured facts

E03 quantizer-conformant full sweep: 20/20 OK, no OOM. Both paths share the same absmax/7 quantized cache values, block table, geometry and BF16 query/output dtype. The historical v1 is preserved verbatim. Its physical tensor axis order differs from v2; conversion is untimed. Source commit: `a97508beb766a9a7a57f508399fb6a6640614b69` (clean).

Corrected raw CSV: `kernel/results/E03_split_scaling/20261014T205041Z_84c662de/split_scaling.csv`. Worst shared-cache output relative error: 0.000154668. Comparator gate threshold is 0.01; E02 threshold 0.002 remains unchanged.

V2/v1 aggregate-throughput ratio:

| Context | B1 | B4 | B8 | B16 | B32 |
|---|---:|---:|---:|---:|---:|
| 8192 | 5.052 | 5.400 | 2.897 | 1.802 | 0.983 |
| 32768 | 22.489 | 7.030 | 3.603 | 1.925 | 1.046 |
| 65536 | 21.868 | 7.192 | 3.827 | 1.970 | 1.061 |
| 131072 | 22.310 | 6.206 | 3.209 | 1.625 | 0.867 |

Batch scaling efficiency is TPS(B32)/(32*TPS(B1)), not a performance pass/fail gate:

| Context | v1 efficiency | v2 efficiency |
|---|---:|---:|
| 8192 | 0.609 | 0.119 |
| 32768 | 0.815 | 0.038 |
| 65536 | 0.947 | 0.046 |
| 131072 | 0.947 | 0.037 |

E05 full sweep: 20/20 OK, no OOM. FP16 ragged/random-paged correctness error 0.00020218 <= 0.002. Both paths use the same 1024-token split, 16-token tile, online-softmax and second-stage merge, four warps/two stages. FP16 query/output, FP32 accumulators. Source commit: `6ad8df898c0cb62a7ebae63825ca7abf2ae66405` (clean).

Raw CSV: `kernel/results/E05_same_memory_budget/20261014T204746Z_a139e189/matched_budget.csv`.

INT4/FP16 aggregate-throughput gain at matched KV budget:

| Context | FP B1 / INT4 B3 | FP B2 / INT4 B7 | FP B4 / INT4 B14 | FP B8 / INT4 B28 |
|---|---:|---:|---:|---:|
| 8192 | 1.419 | 0.805 | 0.551 | 0.360 |
| 16384 | 1.167 | 0.550 | 0.361 | 0.248 |
| 32768 | 0.636 | 0.362 | 0.249 | 0.255 |
| 65536 | 0.389 | 0.265 | 0.269 | 0.258 |
| 131072 | 0.243 | 0.240 | 0.232 | 0.219 |

Measured INT4/FP16 KV-byte ratios are 0.84375 for B1/B3 and 0.984375 for the other pairings. Direct buffer allocations demonstrate 3.0x or 3.5x resident batches within the specified KV budget. Maximum GPU/serving capacity was not measured. Workspace sizes are separate CSV fields and are not included in KV-budget matching.

## System interpretation

On this H200, v1 does not show an increasing-batch throughput collapse in the measured matrix. It starts severely underutilized at small batches; adding requests fills the device. Split-KV improves small-batch long-context utilization, but its advantage declines as batch increases and is lost at 128K/B32. Thus the data do not demonstrate a universal resolution of historical high-concurrency collapse.

KV compression converts into buffer residency, but the current packed INT4 implementation does not convert it into universal throughput. Benefit was measured only at FP B1 for 8K/16K. At longer contexts or larger batches the matched FP16 core is faster despite reading more KV bytes. GQA reread cannot explain these Q/KV=1 measurements because no KV head is shared between query heads. Both E05 paths have the same bounded split-context traversal, so context serialization alone does not identify the difference.

## Unverified hypotheses

The INT4 register unpack/dequant path, redundant nibble/scale loads, reduction work and excessive split scheduling may limit throughput. Their individual contributions are not profiled. E03 also includes a physical-axis-layout difference, so its gains cannot be attributed exclusively to split scheduling. No maximum capacity, real-model quality, end-to-end latency, GQA=4 effect, or production-engine comparison was measured. Single sweep with fixed path order does not establish confidence intervals or clock/order robustness.

## Next optimization candidates

1. A same-layout serial control and profiler counters to separate scheduling from layout and dequant costs.
2. Profile unpack/dequant, redundant loads, stage-1 versus merge time before choosing any optimization.
3. Examine batch-aware split policy in a new frozen experiment; do not retrospectively tune these results.
4. Validate Q/KV=4 and, later, a GPU FWHT plus real-model serving evaluation. E04b remains BLOCKED.

## Direct answers

**A. Did split-KV solve v1 high-batch collapse?** No general solution is established. It strongly improves low-batch long-context core throughput, but high-batch advantage shrinks or reverses; v1 collapse itself was not reproduced on H200.

**B. Is matched-KV-budget INT4 better than FP16?** It provides measured 3.0–3.5x KV-buffer batch residency. Throughput improves only in the 8K/16K FP-B1 region and is worse elsewhere in this matrix.

**C. What long-context serving claim is supported?** Correct paged synthetic attention cores through 128K, low-batch split-KV gains against the archived serial core, and matched-KV-buffer residency. Actual long-context serving, model quality and E2E acceleration remain unverified.

## Provenance and exclusions

All smoke/full run directories, CSVs, run.log and MANIFEST.json are preserved. Initial E03 runs at `c4bd484` used arbitrary packed codes/scales; retain them as screening only and exclude them from quantizer-conformant claims. Corrected E03 runs at `a97508b` are the final reported runs. E05 is explicitly protocol revision `E05_core_paged_v1`, not the original E04b-dependent E2E claim. Existing E00/E01/E02/E04a/E07 artifacts are retained. Main was not modified.
