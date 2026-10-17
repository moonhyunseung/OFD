# E14 plan review and execution contract

Reviewed plan: e3e7685f2b95bc11426b947948df43cdaec1ee82. This review adds executable definitions without changing the historical plan/protocol or E00–E13 kernels/results.

## Assessment

The operator-first direction is justified: E10–E12 identify properties of one executor, not a fundamental compression law. Separate exact equivalences, observable approximation error, runtime feasibility and generated-instruction evidence. E14 can reject mechanisms before investing in a new kernel.

The current design is scientifically promising but its broad protocol alone is not an executable acceptance test. It leaves samples, rank tolerances, fitting/evaluation separation, radix representation, truncation semantics and task-level acceptance limits unspecified. execution.json freezes the first observational implementation. It deliberately does not invent a universal E15 acceptance threshold.

## Mathematical checks and failure modes

1. rank(Q)<=6 at one step follows from Q having six rows. Instantaneous nullity is not empirical compression evidence. Use stacked-window spectra, multiple absolute/relative rank tolerances and past-only fitted basis on future steps. Report raw singular values; numerical rank is not exact symbolic rank.
2. A same-step SVD can concentrate a six-dimensional query into one 32-wide block by construction. It is an oracle upper bound. Recomputing K in that basis costs a full-cache projection; neither persistent storage nor work reduction follows. The separately frozen past-window basis is fit only on steps0–3; held-out starts at4. A 64-step capture has only a 60-step held-out horizon; multiple independent sequences and later separated horizons are needed for a stability claim.
3. H32 preserves every existing 32-block L2 energy exactly. It may change coherence/quantization/plane counts inside a block, but cannot improve energy allocation between the eight blocks. This is an analytic negative control. Full H256 can help quantization yet weaken coordinate/block concentration; these are different objectives.
4. Current OFD uses the same unrandomized normalized H256 as the full-H256 candidate. Record it as an alias, not independent supporting evidence. No randomized signs or newly tuned basis is silently substituted.
5. Scaled centered score error is (Q Delta K^T)/sqrt(d), centered across all tokens. Also report probability TV, K-only output, V-only output and joint output. A small unscaled reconstruction norm or large energy capture alone is insufficient. Near-zero reference norms retain absolute error and denominator.
6. The V support bound delta*max||v|| applies to removal without renormalization. Renormalizing changes the operator and requires a different bound. Use all-token softmax, record actual dropped mass and verify the bound numerically.
7. An orthogonal transform followed by FP16 rounding is not an exact change of basis in floating point. First verify FP64 dual-action equivalence, then record FP16 transform-rounding error separately from binary-plane truncation error. Signed magnitude binary planes use exact FP16 lattice2^-24, including subnormal/extreme values. Nonempty planes and contiguous bit span differ; neither equals measured MMA instruction count.
8. The group-scale matrix is Lx8. Its rank<=8 is also dimensional, not a compression discovery. Its empirical spectrum may suggest an approximation only after operator-error and metadata/reconstruction costs are included.
9. K reduction width32 and V reduction over tokens are different circuits. V scales vary per token. Do not reuse the K tile-count story unchanged for V. Static tile counts are candidate models, not emitted-instruction claims; no new kernel or speedup is measured.
10. Moving V's inverse into W_O is exact only when the maps are adjacent or intervening maps are handled. A dynamic coordinate gate generally does not commute with a dense rotation. Observed target code Qwen3_5Attention applies sigmoid(gate) between attention output and o_proj. The implementation captures this native gate, holds it fixed in the local perturbation diagnostic, and projects the resulting error through original W_O slices. This is a KV-group contribution, not propagated complete-model error. Direct absorption into W_O without handling the intervening gate is not justified.

## Frozen sampling and scope

Full capture: contexts2048/8192/32768, four existing E09 evaluation streams (prose/code/math/long_document), layers3/19/43/63, all four KV heads,64 teacher-forced decode steps, B1. Reuses pinned E09 model/data revisions and chunked native prefill. These streams concatenate source documents using the existing E09 recipe; they are not independent rows just because multiple steps/heads/layers are recorded. Existing datasets do not by themselves define broad distributional generalization.

Smoke: one prose stream, context128, layer3/head0,8steps. Tests mechanics only. No E14 structural or E15 feasibility claim from smoke. Full checkpoint shard SHA256 hashes are required for full capture; smoke identifies pinned revision but skips the expensive shard rehash and remains smoke-only.

Current-query block selection can be computed from an available query, but its selection/packing costs are unpriced. Skipping reads of selected cached blocks is different from persistently deleting those blocks; no cache-capacity claim follows. Per-query-head output errors accompany group norms so a dominant head or cancellation does not hide a weak head.

Raw Q/K/V retain initial all-token K/V plus per-step appends and post-RoPE Q. Raw projection weights stay in the server run directory and are not redistributed in git. Hashes, exact model revision/data manifests, raw numerical tables and tools are archived; captures can be reused for analysis without rerunning the model.

## Numerical analysis outputs

- rank.csv: window spectra, stable rank, energy-entropy effective rank and all tolerance ranks.
- drift.csv: principal angles/chordal distance by lag and subspace dimension. Degenerate spectral cutoffs require caution; full spectra accompany the interpretation.
- blocks.csv: fixed-prefix versus current-query selected block frontiers; distinguish held-out and oracle observations.
- geometry.csv / quantization.csv: coherence/dynamic range, scale spectra, frozen INT4 reconstruction and operator errors.
- planes.csv / plane_frontier.csv: exact finite-FP16 representation and fixed operator-budget frontiers, with transform rounding separated.
- support.csv: all-token probability support, effective support, unrenormalized truncation bound and actual errors.
- circuit.csv: packed/scale bytes, logical contractions, candidate tiles/utilization, transform operations and unpriced terms explicitly marked.
- basis_definitions.json: construction, hashes and alias identities.
- summary.json / FINDINGS.md: per-domain/context/layer/head summaries; no automatic novelty, speed, quality or E15 PASS.

## Run / recover

Use the existing isolated H200 venv and set PYTHONPATH to kernel/src. Commit the source and verify a clean tracked worktree first. One shell block:

```bash
export PYTHONPATH="$PWD/kernel/src"
PY=/home/inugpu/workspace/ofd_v23_f4fvMC/.venv/bin/python
"$PY" kernel/tools/test_e14_math.py --device cuda --out /tmp/e14_math.json && "$PY" kernel/tools/bench_e14_capture.py --smoke --analyze
```

The full capture uses the same command without --smoke. An interrupted analysis reuses its capture via analyze_e14.py RUN --device cuda, followed by summarize_e14.py RUN. Failure status, partial raw files and logs are preserved. Do not run full capture until smoke and output-shape/hash checks pass. No exit/logout/set-e is used.

ETA is not yet measured for this new offline analysis. First smoke establishes elapsed time; full workload is 12 model contexts x4layers x4heads x64steps and must not be scheduled from an unmeasured precise estimate. Raw capture scale is several GB plus projection weights; model and data caches are reused. Analysis is untimed and excluded from all performance comparisons.
