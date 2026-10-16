# E12 low-level findings

Source: 91414cc4ad8eeafe4e4338df25e481649817cd6a. Raw archive: a6ead28a60570ec8e2c4db30165e053026f0feeb.

## Direct observations

All 3,024 timings and 1,008 correctness checks completed on H200. Worst relative L2 2.086e-5; frozen limit .002. Exact Q/KV pointers were preserved across paired capacity/path comparisons. Smoke also passed the independent FP32 oracle, including zero and split-boundary lengths. All raw rows retained.

The disabled guard / original median latency ratio is 1.000022 for INT4 and 1.000059 for FP16. Individual ranges are .96145–1.02887 and .95360–1.04569. Manual process-level review finds no large repeatable implementation drift. Removing the three guards and flag from the new kernel reproduces the original AST.

## Effect sizes

At T=32K/B4, capacity=128K: INT4 g2 decreases 6.946645 to 1.921034 ms (3.615x), INT4 g1 8.723798 to 3.157396 ms (2.763x); FP16 g1 1.569135 to .506538 ms (3.099x). These are overcapacity effects with the same actual KV, not serving speedups.

At T=32K/B4, capacity=32K, all splits active: INT4 g1 increases 2.460916 to 3.153323 ms; g2 decreases 1.954552 to 1.919423 ms. FP16 g1 .498276 to .500076 ms. At capacity=T+16 with actual length unchanged, INT4 g1 increases 2.480832 to 3.825784 ms. The guard is not an unconditional optimization.

Ragged lengths [0,1023,1024,1025], capacity8192, g1: INT4 .575441 to .244395 ms (2.354x), FP16 .114250 to .072621 ms (1.572x). At capacity2048 INT4 regresses .232100 to .240038 ms.

## Supported interpretation

Completely inactive split tile loops impose avoidable cost in both representations. Padding can change latency substantially while actual active KV is fixed. The enabled branch also changes active-path performance; INT4 g1 regressions prevent a universal recommendation. Keep this experimental and opt-in.

## Rejected interpretation

These measurements do not establish HBM, integer unpack, spill, or occupancy as the exact bottleneck, nor a general INT4-specific explanation. The disabled control separates copying from enabled-branch effects, not their microarchitectural mechanism.

## Remaining uncertainty and model control

Uniform E09 contexts are multiples of1024. Append occurs before attention; capacity T+16 and actual T+1 already require the same split count. There are no entirely inactive splits during timed decode. Model confirmation is therefore a branch-cost control, not an opportunity to realize the 4T padding gain. Fixed groups INT4g2/FP16g1 were chosen in E10 and remain frozen. Four prespecified shapes, three shuffled rounds, all64 layers and original data/model revisions. No retuning to hide regression.
