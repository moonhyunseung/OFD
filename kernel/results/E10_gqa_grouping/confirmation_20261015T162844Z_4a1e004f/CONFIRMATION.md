# E10 whole-model grouping confirmation

Source 27092f57dfa2e06d4aa56a0977bc431b52bc9af3; complete=True; rows=12/12; smoke=True.

Fixed grouping policy was committed before any model run. FP16 best fixed g=1, so FP16_gbest repeats the g1 execution as a timing control; it is not a distinct optimized kernel. INT4 uses plain P with g=2. No calibration or layer selection is performed.

| Context | B | Representation | Fixed g | g1 ms | Fixed-g ms | Paired speedup median | Paired speedup range |
|---|---:|---|---:|---:|---:|---:|---|
| 128 | 1 | FP16 | 1 | 91.2047 | 52.5919 | 1.0110 | 1.0018 .. 1.7445 |
| 128 | 1 | INT4 | 2 | 53.6459 | 91.5450 | 0.9106 | 0.5860 .. 1.0061 |

All64 layers, online append, LM head and teacher-forced decode are timed using unchanged E09 wall/event timing. Full runs use 16 decode steps, four warmup and twelve timed; smoke uses four steps and one warmup. Prefill, setup, sampling and serving scheduling are excluded. Online capacity is T+decode_steps, Ns=ceil((T+decode_steps)/1024), unlike standalone low-level exact T. Fixed-g comparisons use the same capacity, layout, bytes and model/data revision.

Native GDN reference fallback observed: True. This does not establish performance with optimized native fast paths or production serving. Rounds share one model process; small speed differences are not independent-process confirmation. FP16 gbest/g1 is an identical-execution control: its observed round spread must be considered before attributing small whole-model effects to grouping. No native logit reference is collected in this speed-only confirmation; finite logits are required and online subgroup equivalence is checked separately in the smoke.

Failures/skips: []
