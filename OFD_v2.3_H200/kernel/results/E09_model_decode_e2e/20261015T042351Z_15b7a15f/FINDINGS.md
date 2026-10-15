# Whole-model decode findings — H200, Qwen3.8-27B

Completed 917 measurements: 8 controls, 384 calibration (64 policies x 2 domains x 3 rounds), 180 held-out evaluation and 345 crossover. 128K/B8 was skipped by HOST_TEMPLATE_LIMIT, not GPU OOM. Failed append validation attempts and the successful smoke are retained separately.

## Direct observations

Four held-out domains (prose/code/math/long document), contexts 2K/8K/32K, B1: geometric means of the 12 shape/domain paired-median throughput ratios versus native FP16 are P 0.965675, KV 0.947156, MIX 0.951010. Versus matched paged FP16: P 0.991818, KV 0.972438, MIX 0.977623. MIX/P is about 0.9848; the calibrated heterogeneous rotation policy did not improve held-out speed over all-P.

Median synchronized full-model decode latency (ms/token):

| Context | B | Native FP16 | Paged FP16 | Plain INT4 P | Rotated INT4 KV | MIX |
|---|---:|---:|---:|---:|---:|---:|
| 32K | 1 | 59.821 | 56.428 | 55.815 | 57.502 | 58.243 |
| 32K | 4 | 55.843 | 55.324 | 71.648 | 71.767 | 71.640 |
| 128K | 1 | 54.118 | 55.810 | 69.197 | 69.298 | 69.169 |
| 128K | 4 | 54.753 | 61.598 | 184.133 | 184.260 | 184.166 |

P beat native in all three paired rounds at 32K/B1 (median paired ratio 1.0556) and 2K/B2 (1.0540); MIX at 2K/B2 (1.0411). These isolated wins do not establish a monotonic context crossover: P loses at longer contexts. Ratios are medians of paired round ratios, not ratios of independently computed latency medians. Round variability is retained in raw data and min/max fields; small wins need confirmation.

At 128K/B4, actual full-attention KV buffers shrink from 34,363,932,672 to 9,664,856,064 bytes (3.55556x), while total allocated GPU memory shrinks from 89,639,366,144 (paged FP16) to 64,940,289,536 bytes (~1.38x). P throughput is 21.723 aggregate tokens/s versus native 73.055 and paged FP16 64.937. Compression is real; a throughput or maximum resident-batch gain is not established here.

The frozen speed-selected policy is P at layers 3,7,15,19,27,31,55,63; K at 11,23,39; V at 43,47,51,59; KV at 35. This is independent K/V rotation selection with both buffers INT4, not independent precision selection. The 4L main-effect screen is not an exhaustive interaction search.

## Interpretation

The current INT4 implementation is not a general real-model decode speed optimization. At high batch/context the three INT4 policies converge to nearly identical slow latency, making rotation policy a weak speed lever in that region. This is compatible with a shared packed-core bottleneck, but does not isolate reread, unpack/dequant or scheduling costs without profiling. Weight and linear-attention work dilute KV-only memory savings.

## Unverified / next work

No optimized production-serving claim: native GatedDeltaNet uses its PyTorch reference fallback because optional fast-path packages are absent. All paths share that fallback. Prefill, sampling, networking and continuous batching are excluded. Only short teacher-forced decode is measured; logits/NLL do not prove downstream generation quality. Batches repeat independently allocated inputs, without prefix sharing. HOST_TEMPLATE_LIMIT is a CPU setup constraint and cannot identify GPU residency limits.

Next candidates are profiling the shared packed core under actual Q/KV=6,D=256 geometry, validating results with native linear-attention fast paths, and a separately frozen capacity experiment that builds each representation without a large CPU FP16 template. Do not change the present protocol or claim results for those unrun configurations.

## Provenance

Measurement source: 2b3585a6b5354c47f30b39c5d87e1d719f682eb2 (clean at launch). Analyzer source: ad8ccbc269e2e4b3d2241d6ae10935732159c176. Model/dataset revisions, seeds, environment and commands are in MANIFEST.json and data_manifest.json. Raw CSV/log/manifest preserved. Existing main is unchanged.
