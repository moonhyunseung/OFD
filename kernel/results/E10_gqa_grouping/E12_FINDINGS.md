# E12: inactive-split loop guard and complete-model control

## Scope

Matched INT4 and FP16; P only; no rotation, new quantizer, nibble/layout changes, split/tile/warp/stage tuning, merge changes or policy search. E00–E11 unchanged. Low-level source 91414cc4ad8eeafe4e4338df25e481649817cd6a; model source af9ee2e1b08b6e35e96ce04fb56bc54e333db059. Both committed clean before claim-eligible runs.

## Low-level direct measurements

3,024 timing blocks, 1,008 checks; worst relative L2 2.086e-5, limit .002. Three independent processes and three shuffled rounds. Exact Q/KV pointer reuse across capacity/path controls. Random original E10 cache; smoke also includes independent FP32 oracle on ragged/boundary cases. Error checks compare the same KV representation; these are not measurements of INT4 model-quality loss relative to unquantized FP16.

| T/B | Capacity | Variant | Original ms | Guard ms | Paired speedup median |
|---|---:|---|---:|---:|---:|
| 32K/B4 | 128K | INT4 g2 | 6.946645 | 1.921034 | 3.6151x |
| 32K/B4 | 128K | FP16 g1 | 1.569135 | .506538 | 3.0992x |
| 32K/B4 | 32K | INT4 g1 | 2.460916 | 3.153323 | .7804x |
| 32K/B4 | 32K+16 | INT4 g1 | 2.480832 | 3.825784 | .6485x |

The copied-but-disabled path matches original median ratios INT4 1.000022/FP16 1.000059. Enabled-guard g1 regressions are retained. The optional guard is not promoted to default.

## Causal answers across E10–E12

Q1: Useful Stage-1 work alone is insufficient. E11 fixes actual Q/KV and changes capacity, increasing latency 3.55x INT4 /3.15x FP16. E12 intervention removes much of this excess without changing active KV. Launched masked work, representation and compiler/execution effects matter as well.

Q2: No systematic independent high-batch penalty is supported in the tested equal-work triplets. E11 also repartitions fixed total heads/KV/work from B1/Hkv16 to B16/Hkv1: g1 ratios .9522 INT4/.9829 FP16, not an adverse collapse. This repartition changes head shape/strides with batch, so pure batch causality is not identified and batch independence is not proven.

Q3: E10 supports representation-dependent grouping response at high work (GI positive, GF negative). It does not uniquely identify packed load/unpack/dequant repetition. E11 changes D with unchanged active KV; E12 changes active-path speed merely by enabling a guard. Resource/scheduling interactions remain live. D is not instruction time or saved HBM traffic.

## Interpretation limits

These experiments rule out several single-factor explanations within this matrix; they cannot make every possible dependency claim impossible. No exact HBM, spill, occupancy or instruction percentage is measured. Profiler attribution remains future work. No production-serving or broad full-model speedup claim follows from a synthetic 4T capacity gain.

## Complete-model direct measurements

Smoke 12/12 rows and 16 online equivalence checks pass (worst relative L2 2.3542e-5). Full 48/48 rows pass, finite logits, three shuffled rounds in one process. No rows dropped. All64 model layers, online append, LM head and teacher-forced decode are retained; prefill/cache construction is untimed. Original model/data revisions and reference PyTorch GDN/conv fallback remain. No native-logits quality reference is measured in this speed-only follow-up.

The capacity is T+16, with Ns=9,33,129 for T=8K,32K,128K. Actual length is T+s+1 after append, so every allocated split is already active at the first step. Consequently this is an active-path overhead control, not a test of removing whole empty splits in serving. Fixed INT4g2/FP16g1 were frozen in E10 and before this run. The legacy model path uses original OnlineKV dispatch (including legacy FP16g1); guard dispatch uses the E12 generalized wrapper. E10 g1 compatibility and E12 online checks apply; the full-model test alone does not isolate wrapper cost.

| T | B | Rep/g | Legacy median ms | Guard median ms | Paired speedup median | Paired range |
|---|---:|---|---:|---:|---:|---|
| 8192 | 1 | FP16/g1 | 52.6567 | 52.1144 | 1.016302 | 0.940815–1.019727 |
| 8192 | 1 | INT4/g2 | 52.8917 | 53.2555 | 0.993715 | 0.989319–0.999701 |
| 32768 | 4 | FP16/g1 | 51.6811 | 51.9025 | 1.003109 | 0.992818–1.008438 |
| 32768 | 4 | INT4/g2 | 63.4494 | 62.6734 | 1.012653 | 1.012174–1.012845 |
| 131072 | 1 | FP16/g1 | 52.7540 | 53.1981 | 0.991653 | 0.975760–1.009139 |
| 131072 | 1 | INT4/g2 | 60.2569 | 59.1555 | 1.018888 | 1.018619–1.019315 |
| 131072 | 4 | FP16/g1 | 61.6728 | 61.1666 | 1.008633 | 1.008107–1.010192 |
| 131072 | 4 | INT4/g2 | 154.6220 | 151.7925 | 1.018691 | 1.017332–1.018715 |

Ratios are paired within round; ratios of marginal medians can differ. The three rounds are not three independent processes. FP16 8K/B1 has a broad .9408–1.0197 range; smoke INT4 retains one .5448 ratio. No outlier exclusion or posthoc threshold. Larger INT4 cases show small observed 1.3–1.9% speedups, not the synthetic 3.6x gain. Since no split is wholly inactive, these effects cannot be attributed to skipped inactive loops; compiler/execution and wrapper effects remain possible. This is not a claim of equivalence, statistical significance, or a production serving speedup. INT4 remains slower than FP16 at every measured complete-model shape.

At128K/B4, guarded INT4 median151.7925ms versus guarded FP16 61.1666ms. KV allocation is 9,664,856,064 versus34,363,932,672 bytes (3.55556x representation compression); per-token starting GPU allocations about64.94 versus89.64GB in round0. These are same-batch comparisons, not a same-memory-budget resident-batch experiment. Model weights/common caches and host templates remain outside the KV compression ratio.

## Warnings and provenance

Full run returns0 with no traceback, but run.log preserves17 CUDA allocator OOM warnings. All occur before the first128K/B4 result row; the log does not mark each warning's exact phase, so do not label this run warning-free or assign every warning to one shape or timed decode. No shape, allocation strategy, precision or protocol was changed to recover a favorable result. Native GDN/conv reference fallback makes absolute timings specific to this harness; optimized serving backends are untested.

A postprocessing command first attempted the absent executable `python`; retry with `sys.executable` succeeded. This did not affect measurement workers. See postprocess_recovery.log.

Low-level raw archive a6ead28a60570ec8e2c4db30165e053026f0feeb; low-level validation and model source af9ee2e1b08b6e35e96ce04fb56bc54e333db059; model raw archive8270c7221a268a1de193dbd341e4c2ae245815b1. Per-run manifests retain exact source/hardware/library/seed/protocol. ARCHIVE_HASHES.json retains original-byte and archived-LF text hashes; text export normalizes CRLF, not numeric content. Existing history/main unchanged.

## Supported next action

Keep guard opt-in. Quantify Stage1 versus merge and profile the frozen active INT4g1 regression and high-workg2 path, with matched FP16 controls. This would attribute mechanisms after causal interventions; do not add nibble, warp, split, precision, rotation or layout tuning to this completed comparison. There is no evidence here to enable the guard globally or advertise a broad serving win.
