# E10 — matched subgrouped GQA causal test

Frozen design: [protocol.json](protocol.json). Source branch begins at `9a82b02d9ce6dd805d45c41229258c9a88c1f496`. Existing E00–E09 protocols and evidence are unchanged.

From a clean exact source checkout with the existing H200 venv:

```bash
PYTHONPATH=kernel/src python kernel/tools/bench_e10_grouping.py --smoke
PYTHONPATH=kernel/src python kernel/tools/bench_e10_grouping.py
```

Run smoke first and inspect correctness and generalized/legacy g1 latency ratios before the complete sweep. A full run launches three sequential independent Python processes. Each shape has three shuffled rounds of 8 generalized execution variants plus legacy INT4/FP16 g1 controls. Each timed block uses 10 warmups and 50 repeats with the existing `cuda_ms`, Stage 1 and unchanged merge included. Compile and validation are outside timing. Both representations coexist during the low-level run; actual cache layout/quantizer/nibble decoding/scale dtype are unchanged. FP16 pages and their quantized counterparts share physical page indices and sequence lengths. This is same-shape comparison, not same-memory-budget.

The generalized kernel is adapted directly from existing `grouped_kernels.py`; a compile-time representation branch replaces only the FP16 load/dequant segment. Logical groups 3/6 use padded query axes 4/8 with masked loads/stores because Triton arange requires powers of two. This padding is identical for INT4 and FP16. Existing full grouped INT4 also receives this necessary non-power-of-two correctness extension, without changing the original q-head kernel, merge or old power-of-two grouping shapes. Its g6 output is checked separately on every shape.

Standalone capacity is exactly T: `Ns=ceil(T/1024)`. E09 online capacity is T+16 and must be recorded separately if confirmation is reached. Stage-1 count is `B*Hkv*(6/g)*Ns`; W1 is `B*24*Ns`. Equal-work groups here have exactly equal W1, but also equal B*T/cache footprint. Similar latency supports a work/footprint axis, not proof that program count alone causes slowdown. Systematic differences within a group support an effect of the combined batch/context geometry, not isolated batch causality: holding W1 fixed necessarily changes context.

No new performance threshold will be chosen after results. Raw g1 ratios and process variation must be reviewed before causal interpretation or full-model confirmation. GPU attribution (HBM traffic, spill, unpack instruction costs) requires later profiling. Full model stays gated until low-level numerical/compatibility review passes; no E09 policy search is rerun.

Initial smoke at 5cdaeb1 passed numerical checks but generalized rank-3 g1 cost differed from legacy (INT4 ~1.06x, FP16 ~1.37x at 2K/B1). Full sweep was withheld. Generalized g1 now specializes to an exact copy of each original scalar q-head body inside the same compile-time kernel; g>1 retains matched subgrouping. A fresh smoke must pass before continuation. Failed-parity evidence is retained, without changing timing or kernel configuration.
