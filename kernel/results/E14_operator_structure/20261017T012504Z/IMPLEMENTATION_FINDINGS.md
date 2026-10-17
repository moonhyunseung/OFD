# E14 implementation and numerical readiness

## 1. Evaluation

The post-E12 operator-first direction is appropriate. It asks for a mathematically identifiable work reduction before another kernel optimization. The important scientific question is persistent/usable structure under real query evolution, not the tautology rank(Q)<=6 at one decode step.

The executable plan must separate:

- same-step SVD oracle from a basis fit on past queries and evaluated on future queries;
- H32 intra-block flattening from between-block energy concentration (H32 cannot change the latter);
- exact basis algebra from finite-FP16 transform rounding and quantizer error;
- feature reduction from token support truncation;
- logical circuit accounting from actual emitted instructions and speed;
- local operator error from propagated model quality.

A concrete target-model condition was observed: Qwen3_5Attention applies a dynamic sigmoid coordinate gate between attention output and o_proj. For one query head, the local map is y G_t W_O. A dense V basis B cannot be absorbed into a fixed W_O merely by writing B^-1 W_O while leaving G_t unchanged; that requires commutation or explicitly handling the conjugated gate. The capture records the original gate and analysis holds it fixed. No whole-model quality or free weight-absorption claim follows.

These are refinements of execution and interpretation; historical protocols and E00–E13 code/results remain unchanged. No publication novelty or general compressed-serving benefit has been established by E14 smoke.

## 2. Implemented experiment

Capture source: ac06eb60a03c2fb7bfd74d19b2a0b7fcb15b712b, clean tracked worktree before capture. Analysis revision: cc2c9d12034963fb2a52b660a136c8901e7e7943. Raw text archive:18eb2b7da1f5e009b9f59e7f05e3200d92860830.

- bench_e14_capture.py reuses pinned E09 Qwen3.8-27B model/data/prefill. It captures post-RoPE Q/K, original all-token V, per-step K/V appends, native output gates and original projection slices. All64 model layers execute, while only configured full-attention layers/heads are recorded.
- analyze_e14.py produces FP64 rank/drift, block concentration, centered/scaled score and probability/output errors, basis/scale/coherence, exact finite-FP16 binary-plane diagnostics, V support bounds and a static circuit ledger.
- test_e14_math.py independently checks symmetry, constant-logit quotient, H32 invariance, groupwise unsigned rank1/scale identities, FP16 extreme/subnormal plane reconstruction and V support bounds, plus a synthetic analyzer fixture.
- summarize_e14.py retains domain/context/layer/head and fitting/evaluation strata. It summarizes fixed operator-budget plane frontiers without posthoc acceptance thresholds.

Five distinct mathematical basis candidates are implemented: identity, block H32, full H256, frozen first-four-step query SVD and same-step query SVD oracle. Current OFD is a mathematical H256 alias, not another independent candidate or a bitwise-FWHT equivalence claim.

## 3. Executed numerical validation

H200, pinned model revision1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0. Actual smoke: prose evaluation stream, context128/B1, layer3/KVhead0,8 teacher-forced steps. Baseline native sigmoid gates were finite with observed range .0002694129944 to .437255859375. Original PyTorch GDN/conv reference fallback remains; no timing result is claimed.

13 mathematical checks PASS. Native capture, original analysis and refined analysis all completed. Post-validation confirms raw state hashes, initial cache length128, eight Q/gate/append records, finite states, gate range[0,1] and finite CSV numerical fields. Archived text hashes match the validated LF export.

| Table | Rows in refined analysis |
|---|---:|
| rank |15|
| drift |51|
| blocks |640|
| quantization |40|
| geometry |120|
| planes |824|
| plane_frontier |40|
| support |240|
| circuit |8|

Total1,978 diagnostic rows derive from only eight decode positions. They are not1,978 independent samples. The summary contains160 block strata.

One mechanical check illustrates why oracle/future separation matters. On held-out steps4–7, fixed-prefix first-block query energy capture ranges:

| Basis | Captured energy range |
|---|---|
| identity |.237152–.288597|
| block H32 |.237152–.288597|
| full H256 |.103456–.136185|
| past-window SVD |.463298–.528970|
| current-step SVD oracle |1.000000–1.000000|

H32 matches its analytic invariance; current-step SVD's100% is expected by construction. The past-window result is only a smoke observation from four future steps, not evidence of persistent structure across model layers, long contexts or prompts. Energy alone does not establish acceptable score/output error or profitable hardware elision. All operator-error frontiers are retained in CSV.

Refined analysis explicitly rounds transformed K/V to FP16 before the frozen encoder and separates transform-rounding from quantizer-only error. Its bit-plane work ledger conservatively models separate positive/negative unsigned contractions. Unique logical payload bytes are not HBM traffic; repeated-plane reuse, conversion lowering, SVD/reprojection and legal runtime basis costs remain unpriced. No PTX/SASS or latency claim exists.

## Readiness and remaining work

Status: IMPLEMENTED_AND_SMOKE_VERIFIED. Scientific gate: SMOKE_ONLY. Full matrix NOT RUN.

The configured full observational capture spans2048/8192/32768 contexts, four existing E09 evaluation streams, layers3/19/43/63, all four KV heads and64 steps. It hashes checkpoint shards, reuses model/data caches and writes unique results. First four steps fit the past-only basis;60 future steps remain. Report worst head/sequence errors alongside aggregate norms and never treat heads/steps as independent prompts.

A full run remains an observational pilot. E15 acceptance still requires a declared application-level error budget, legal reusable basis/transform placement, complete work accounting and generated-code verification. A qualitative useful-energy threshold is insufficient to promote a design. The four-step fitted basis also carries an unpriced full-cache reprojection cost.

Raw Q/K/V/gate tensors and model projection weights are retained on the H200 under this run's capture directory. They are not redistributed in git. The archive includes manifests, capture index and hashes, data provenance, raw numerical tables, basis definitions, summaries, validation and logs. It is reproducible by rerunning the pinned capture, not by loading uncommitted weights from GitHub.

The initial source af53b10 passed12 mathematical checks but its capture was blocked before model execution because generated results were untracked. The scoped E14 results ignore fixed this without relaxing the clean check; both attempts are retained. A text-export relative/absolute path mismatch was corrected after numerical validation; recovery log retained. Initial analysis under capture/ is preserved; analysis_v2/ is the refined output. No rows were selected/deleted to improve a result.

See IMPLEMENTATION_REVIEW.md for algebra, sampling, failure modes and one-block launch/recovery commands; execution.json is the frozen implementation profile. No user action is needed to recover this completed smoke.
