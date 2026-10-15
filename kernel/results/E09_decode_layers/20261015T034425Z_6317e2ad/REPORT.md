# Qwen3.8-27B decode layer replay

Source: `fe6c40827b0dabf0df8a217802fc6fb805b43b63`; run `20261015T034425Z_6317e2ad`.

Prefill excluded; 16 full-attention layers, four domains, fixed calibration/evaluation partition. Other 48 Gated DeltaNet layers remain native. Policy chosen from calibration error only; all figures below use evaluation.

Policy counts: {'KV': 16}. Mean attention-output relative error: P=0.0876522, selected=0.0709842. Geometric mean matched paged FP16/selected-INT4 latency ratio=0.310576.

| Layer | Policy | P error | Selected error | Selected/P error | FP16/INT4 latency |
|---:|---|---:|---:|---:|---:|
| 3 | KV | 0.054737 | 0.044015 | 0.8041 | 0.3158 |
| 7 | KV | 0.07971 | 0.06634 | 0.8323 | 0.3091 |
| 11 | KV | 0.078745 | 0.070697 | 0.8978 | 0.3100 |
| 15 | KV | 0.090709 | 0.080066 | 0.8827 | 0.3101 |
| 19 | KV | 0.088314 | 0.076222 | 0.8631 | 0.3122 |
| 23 | KV | 0.083851 | 0.073409 | 0.8755 | 0.3098 |
| 27 | KV | 0.085448 | 0.07401 | 0.8661 | 0.3114 |
| 31 | KV | 0.090236 | 0.073185 | 0.8110 | 0.3120 |
| 35 | KV | 0.095713 | 0.07389 | 0.7720 | 0.3109 |
| 39 | KV | 0.096136 | 0.078538 | 0.8170 | 0.3122 |
| 43 | KV | 0.090535 | 0.068472 | 0.7563 | 0.3114 |
| 47 | KV | 0.10532 | 0.082796 | 0.7861 | 0.3110 |
| 51 | KV | 0.11472 | 0.090324 | 0.7873 | 0.3079 |
| 55 | KV | 0.12229 | 0.08949 | 0.7318 | 0.3089 |
| 59 | KV | 0.066489 | 0.047251 | 0.7107 | 0.3079 |
| 63 | KV | 0.059478 | 0.04704 | 0.7909 | 0.3088 |

Direct measurements concern per-layer attention replay on native model decode states. The selected policy is replay-evaluated, not installed in the model. Total model latency, online KV append, generated-text quality, perplexity and resident serving capacity are unmeasured. One packed stream per domain/partition and nested context prefixes limit statistical generalization. GPU events use the inherited host-launch timing contract and include possible launch starvation. No hardware-counter diagnosis is implied.

System interpretation: lower selected error demonstrates held-out attention-output improvement only where measured. It does not establish downstream task quality or full-model speedup. Next required step is online packed-cache integration and an actual policy-conditioned decode comparison.

Metric definition: INT4 attention error is ||INT4-FP16||_2 / (||INT4||_2 + 1e-12), inherited from bench_gpu.rel; native-baseline error uses the paged FP16 output in the denominator. Baseline and INT4 use the same B=1, post-RoPE head geometry, randomized page table, split/tile/warps/stages. Cache construction and quantization are outside the timed region; query and output GPU FWHT are inside. Native FP16-model weights are converted to FP16 for both paths. KV bytes exclude weights, Gated DeltaNet state, workspaces and allocator overhead. The per-layer mean policy is KV for all 16 layers: this run does not establish a benefit from heterogeneous layer policy.
