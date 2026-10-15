# E10 GQA grouping causal sweep

Measurement SHA: 6ef2a73ec39f33743cc36ccc49bb93b2c373b8e8; complete=True; correctness=True; rows=2160/2160.

## 1. Direct observations
2160 timed blocks and 648 numerical comparisons completed without OOM; maximum relative L2 1.65262e-5. Each shape has nine process/round blocks; repeats inside a block are not independent experimental units. INT4 generalized/legacy g1 median ratio 0.999995; FP16 1.000591. FP16 has one isolated 1.13865 at 2K/B1/process2/round0; the other 215 ratios span 0.989181..1.010846, with no repeated initial-smoke-like g1 drift. No row is dropped. See VALIDATION.json, lowlevel_summary.csv and g1_sanity.csv.

Generalized/legacy g1 ratios: {'INT4': [0.9942569016906012, 1.0062473635782232], 'FP16': [0.9891811186172365, 1.138649794120196]}

## 2. INT4 grouping gain
GI(g)=INT4_g1-INT4_g, in ms. Positive means lower latency. At g2, 14/24 shapes improve; g3 17/24; g6 14/24. At 128K/B4: g1 9.4564ms, g2 7.6412ms, g6 6.0433ms. At 2K/B1: g1 0.2028ms, g2 0.2573ms, g6 0.5680ms: grouping hurts at low work. Across all 24 shapes, best fixed g2 reduces geometric-mean latency from 1.269763ms to 1.154309ms (all process/round blocks equally weighted).

## 3. FP16 grouping gain
GF(g)=FP16_g1-FP16_g, in ms. GF is negative at every shape for g2/g3/g6. At 128K/B4: g1 1.8618ms, g2 3.9382ms, g6 4.9966ms. Thus this subgroup execution is not a generic speed win for FP16. g*=1 in all 24 FP16 shapes, while INT4 g* is 1 in seven shapes, 3 in three, and 6 in fourteen. The best fixed g2 need not be the per-shape optimum; it balances low-work penalties with high-work gains.

## 4. Difference-in-differences D(g)
D=GI-GF, computed within each process/round before aggregation. Raw effects and process mean ranges are retained. No posthoc significance threshold.

| T | B | g | GI median ms | GF median ms | D median ms | D process-mean range ms |
|---|---:|---:|---:|---:|---:|---|
| 2048 | 1 | 2 | -0.054532 | -0.081631 | 0.027039 | 0.024065 .. 0.027204 |
| 2048 | 1 | 3 | -0.177193 | -0.187523 | 0.010326 | 0.007314 .. 0.010793 |
| 2048 | 1 | 6 | -0.365320 | -0.392248 | 0.026997 | 0.023914 .. 0.027037 |
| 2048 | 2 | 2 | -0.066695 | -0.081564 | 0.014836 | 0.014823 .. 0.015457 |
| 2048 | 2 | 3 | -0.196801 | -0.203975 | 0.007178 | 0.006828 .. 0.007620 |
| 2048 | 2 | 6 | -0.391174 | -0.398928 | 0.007668 | 0.007677 .. 0.008251 |
| 2048 | 4 | 2 | -0.026383 | -0.062125 | 0.035632 | 0.028205 .. 0.037034 |
| 2048 | 4 | 3 | -0.159705 | -0.173863 | 0.014150 | 0.007954 .. 0.015037 |
| 2048 | 4 | 6 | -0.389380 | -0.393096 | 0.003540 | 0.000765 .. 0.005876 |
| 2048 | 8 | 2 | -0.011514 | -0.092192 | 0.080946 | 0.080346 .. 0.081620 |
| 2048 | 8 | 3 | -0.098692 | -0.166769 | 0.067862 | 0.066304 .. 0.068246 |
| 2048 | 8 | 6 | -0.334550 | -0.412051 | 0.077346 | 0.076418 .. 0.078237 |
| 8192 | 1 | 2 | -0.027196 | -0.076505 | 0.049309 | 0.046763 .. 0.049668 |
| 8192 | 1 | 3 | -0.160135 | -0.199868 | 0.039642 | 0.035713 .. 0.039784 |
| 8192 | 1 | 6 | -0.389424 | -0.420463 | 0.030905 | 0.027296 .. 0.031386 |
| 8192 | 2 | 2 | -0.011620 | -0.093393 | 0.081484 | 0.080772 .. 0.082208 |
| 8192 | 2 | 3 | -0.098365 | -0.163709 | 0.064893 | 0.064627 .. 0.066142 |
| 8192 | 2 | 6 | -0.334694 | -0.408372 | 0.073164 | 0.072804 .. 0.074608 |
| 8192 | 4 | 2 | -0.002973 | -0.125309 | 0.122374 | 0.120899 .. 0.124336 |
| 8192 | 4 | 3 | 0.159743 | -0.219431 | 0.379338 | 0.378078 .. 0.380056 |
| 8192 | 4 | 6 | -0.015501 | -0.389583 | 0.375055 | 0.371672 .. 0.376519 |
| 8192 | 8 | 2 | 0.268345 | -0.235196 | 0.503574 | 0.502161 .. 0.504472 |
| 8192 | 8 | 3 | 0.287515 | -0.416205 | 0.703777 | 0.701238 .. 0.707157 |
| 8192 | 8 | 6 | 0.461515 | -0.404365 | 0.865911 | 0.864199 .. 0.866494 |
| 16384 | 1 | 2 | -0.011679 | -0.095149 | 0.083555 | 0.082508 .. 0.083639 |
| 16384 | 1 | 3 | -0.099677 | -0.188932 | 0.089327 | 0.088857 .. 0.089553 |
| 16384 | 1 | 6 | -0.334204 | -0.408383 | 0.074136 | 0.073799 .. 0.074905 |
| 16384 | 2 | 2 | -0.002136 | -0.122154 | 0.119935 | 0.119235 .. 0.121258 |
| 16384 | 2 | 3 | 0.155318 | -0.216556 | 0.372260 | 0.368460 .. 0.373340 |
| 16384 | 2 | 6 | -0.015617 | -0.387855 | 0.374127 | 0.371013 .. 0.373978 |
| 16384 | 4 | 2 | 0.268008 | -0.237061 | 0.505069 | 0.502066 .. 0.505284 |
| 16384 | 4 | 3 | 0.299076 | -0.425763 | 0.726380 | 0.722143 .. 0.726605 |
| 16384 | 4 | 6 | 0.471503 | -0.405338 | 0.876528 | 0.875387 .. 0.878354 |
| 16384 | 8 | 2 | 0.495912 | -0.534172 | 1.029316 | 1.029571 .. 1.030287 |
| 16384 | 8 | 3 | 0.523177 | -0.872140 | 1.396266 | 1.393070 .. 1.396846 |
| 16384 | 8 | 6 | 0.883398 | -0.802704 | 1.686476 | 1.682732 .. 1.687299 |
| 32768 | 1 | 2 | -0.014709 | -0.120888 | 0.106228 | 0.105710 .. 0.107352 |
| 32768 | 1 | 3 | 0.143116 | -0.251613 | 0.394796 | 0.388941 .. 0.393908 |
| 32768 | 1 | 6 | -0.025857 | -0.388389 | 0.363139 | 0.360159 .. 0.362782 |
| 32768 | 2 | 2 | 0.265594 | -0.234821 | 0.500738 | 0.495962 .. 0.502991 |
| 32768 | 2 | 3 | 0.300278 | -0.409361 | 0.709640 | 0.707557 .. 0.710083 |
| 32768 | 2 | 6 | 0.480346 | -0.402349 | 0.883778 | 0.882553 .. 0.883186 |
| 32768 | 4 | 2 | 0.505907 | -0.530984 | 1.037252 | 1.036312 .. 1.038258 |
| 32768 | 4 | 3 | 0.542717 | -0.868346 | 1.412182 | 1.410536 .. 1.415048 |
| 32768 | 4 | 6 | 0.899667 | -0.797555 | 1.697226 | 1.693177 .. 1.699912 |
| 32768 | 8 | 2 | 0.875928 | -1.068093 | 1.944109 | 1.940195 .. 1.951251 |
| 32768 | 8 | 3 | 0.921653 | -1.722256 | 2.642886 | 2.648194 .. 2.650981 |
| 32768 | 8 | 6 | 1.635489 | -1.591143 | 3.226184 | 3.221046 .. 3.234938 |
| 65536 | 1 | 2 | 0.276845 | -0.236449 | 0.513411 | 0.510137 .. 0.514576 |
| 65536 | 1 | 3 | 0.336944 | -0.489105 | 0.826390 | 0.824122 .. 0.828041 |
| 65536 | 1 | 6 | 0.508403 | -0.398847 | 0.907200 | 0.906308 .. 0.908877 |
| 65536 | 2 | 2 | 0.537498 | -0.532419 | 1.069091 | 1.068243 .. 1.071264 |
| 65536 | 2 | 3 | 0.580916 | -0.865992 | 1.447766 | 1.440670 .. 1.449270 |
| 65536 | 2 | 6 | 0.945980 | -0.794189 | 1.739332 | 1.735191 .. 1.741378 |
| 65536 | 4 | 2 | 0.950879 | -1.072607 | 2.023419 | 2.019848 .. 2.035618 |
| 65536 | 4 | 3 | 1.006517 | -1.721466 | 2.728544 | 2.720558 .. 2.742133 |
| 65536 | 4 | 6 | 1.716423 | -1.586836 | 3.302741 | 3.297866 .. 3.315439 |
| 65536 | 8 | 2 | 1.686854 | -2.074765 | 3.762888 | 3.760414 .. 3.768222 |
| 65536 | 8 | 3 | 1.791368 | -3.399938 | 5.191388 | 5.180344 .. 5.209756 |
| 65536 | 8 | 6 | 3.266309 | -3.134319 | 6.404340 | 6.391218 .. 6.413168 |
| 131072 | 1 | 2 | 0.552890 | -0.532133 | 1.085461 | 1.080692 .. 1.087597 |
| 131072 | 1 | 3 | 0.637343 | -0.975193 | 1.611344 | 1.607588 .. 1.615000 |
| 131072 | 1 | 6 | 0.994251 | -0.785886 | 1.781112 | 1.776994 .. 1.782288 |
| 131072 | 2 | 2 | 0.988270 | -1.076330 | 2.065139 | 2.065140 .. 2.071608 |
| 131072 | 2 | 3 | 1.045710 | -1.722527 | 2.767308 | 2.767379 .. 2.773612 |
| 131072 | 2 | 6 | 1.759886 | -1.586087 | 3.345885 | 3.347153 .. 3.351070 |
| 131072 | 4 | 2 | 1.815433 | -2.093436 | 3.908637 | 3.895846 .. 3.913756 |
| 131072 | 4 | 3 | 1.924155 | -3.400257 | 5.316548 | 5.304295 .. 5.339367 |
| 131072 | 4 | 6 | 3.413050 | -3.138568 | 6.557405 | 6.525601 .. 6.572702 |
| 131072 | 8 | 2 | 3.463096 | -4.083551 | 7.547979 | 7.541084 .. 7.576630 |
| 131072 | 8 | 3 | 3.662284 | -6.664567 | 10.326379 | 10.330339 .. 10.338893 |
| 131072 | 8 | 6 | 6.590811 | -6.047289 | 12.630775 | 12.620639 .. 12.654283 |

## 5. Equal-work batch analysis
See equal_work.csv. All requested triplets use exact T capacity and have equal W1. Each ratio is paired by process/round against the first triplet member. Context, split count and batch change together; same W1 also means same B*T and KV footprint.

## 6. Supported interpretation
Q1: total work/footprint is a strong organizing axis, but program count alone does not explain all latency. For fixed (T,B,g), INT4 and FP16 have the same Wg and merge, yet different latency and grouping response. At equal W1 the INT4 latency differences are small and track context slightly, not an adverse high-batch ordering. Program count, per-program representation work and resource usage cannot be collapsed to one count.

Q2: no independent high-batch penalty is supported within the two requested equal-work triplets. For g1, relative to 32K/B4, INT4 64K/B2 and 128K/B1 ratios are 1.0183 and 1.0370; FP16 1.0050 and 1.0009. Relative to 32K/B8, INT4 64K/B4 and 128K/B2 are 1.0175 and 1.0265; FP16 1.0039 and 0.9992. Higher B is not systematically worse. This is evidence against that ordering within tested triplets, not proof that batch never matters.

Q3: representation-dependent grouping benefit is supported in the high-work region: GI>0 while GF<0. At 128K/B4/g2, paired median GI=1.81543ms, GF=-2.09344ms and D=3.90864ms (three process-mean D values span 3.89585..3.91376ms). All 72 shape/g D medians are positive, but at low work GI can be negative: positive D alone does not prove INT4 gets faster. In particular D includes the FP16 grouping penalty and must not be described as the amount of INT4 latency removed. Reducing q-head repetition is therefore one useful intervention for current packed execution at high work, consistent with packed load/unpack/dequant reuse, while the exact packed-versus-scheduling mechanism remains unseparated. Grouping reduces program count and within-subgroup repeated representation work simultaneously; D does not isolate instruction cost.

## 7. Rejected interpretation
Rejected within these measurements: a universal grouping speedup, a systematic larger-batch penalty at equal W1, and an explanation using only Stage-1 program count independent of representation. Not established: HBM bottleneck, register spills, integer-unpack percentage, or a production serving win. INT4 remains slower than matched FP16 in these low-level results even after grouping.

## 8. Remaining uncertainty
Process variation, padding lanes (3->4,6->8), shared merge floor, cache residency, and compiler/resource changes remain. Matched FP16 controls do not remove all representation-by-scheduling interactions. No full-model confirmation until manual validity review and a committed fixed policy.

Best fixed g candidates selected by the original protocol and frozen before model confirmation: {'INT4': {'group_q': 2, 'geomean_latency_ms': {'1': 1.2697634863185256, '2': 1.154308963642163, '3': 1.2099174977112421, '6': 1.2668545610278332}}, 'FP16': {'group_q': 1, 'geomean_latency_ms': {'1': 0.3239383600889658, '2': 0.6255667095301797, '3': 0.8820020814404275, '6': 1.0539672073312742}}}

## Completed model follow-up
The frozen fixed policy was applied after validation. See ../README.md and ../confirmation_20261015T163740Z_7bd99095/CONFIRMATION.md for all 96 complete-model rows, paired ranges and limitations. No full-model result was used to revise the fixed policy.
