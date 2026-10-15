# E10 whole-model grouping confirmation

Source 27092f57dfa2e06d4aa56a0977bc431b52bc9af3; complete=True; rows=96/96; smoke=False.

Fixed grouping policy was committed before any model run. FP16 best fixed g=1, so FP16_gbest repeats the g1 execution as a timing control; it is not a distinct optimized kernel. INT4 uses plain P with g=2. No calibration or layer selection is performed.

| Context | B | Representation | Fixed g | g1 ms | Fixed-g ms | Paired speedup median | Paired speedup range |
|---|---:|---|---:|---:|---:|---:|---|
| 8192 | 1 | FP16 | 1 | 53.7419 | 54.0076 | 0.9951 | 0.9889 .. 1.8546 |
| 8192 | 1 | INT4 | 2 | 54.7758 | 54.8891 | 0.9990 | 0.9933 .. 1.0029 |
| 8192 | 8 | FP16 | 1 | 53.9908 | 54.3152 | 0.9990 | 0.9940 .. 1.0141 |
| 8192 | 8 | INT4 | 2 | 59.8377 | 56.7053 | 1.0552 | 1.0526 .. 1.0561 |
| 32768 | 1 | FP16 | 1 | 54.1312 | 53.7868 | 1.0021 | 0.7274 .. 1.0311 |
| 32768 | 1 | INT4 | 2 | 55.0307 | 54.6625 | 1.0073 | 1.0067 .. 1.7503 |
| 32768 | 4 | FP16 | 1 | 54.8849 | 55.2288 | 0.9938 | 0.5354 .. 1.1318 |
| 32768 | 4 | INT4 | 2 | 71.6026 | 63.4615 | 1.1281 | 0.7170 .. 1.5951 |
| 65536 | 2 | FP16 | 1 | 55.1479 | 55.6750 | 1.0015 | 0.9787 .. 1.0024 |
| 65536 | 2 | INT4 | 2 | 69.6595 | 61.2184 | 1.1380 | 1.1376 .. 1.1517 |
| 65536 | 4 | FP16 | 1 | 54.2937 | 54.4584 | 1.0008 | 0.9970 .. 1.0010 |
| 65536 | 4 | INT4 | 2 | 109.6843 | 93.7559 | 1.1699 | 1.1698 .. 1.1699 |
| 131072 | 1 | FP16 | 1 | 53.6506 | 54.6986 | 0.9677 | 0.9578 .. 0.9974 |
| 131072 | 1 | INT4 | 2 | 69.1609 | 60.3940 | 1.1442 | 0.8216 .. 1.1452 |
| 131072 | 4 | FP16 | 1 | 61.6710 | 61.8457 | 0.9989 | 0.9972 .. 1.6749 |
| 131072 | 4 | INT4 | 2 | 184.2367 | 154.6218 | 1.1916 | 1.1910 .. 1.1916 |

All64 layers, online append, LM head and teacher-forced decode are timed using unchanged E09 wall/event timing. Full runs use 16 decode steps, four warmup and twelve timed; smoke uses four steps and one warmup. Prefill, setup, sampling and serving scheduling are excluded. Online capacity is T+decode_steps, Ns=ceil((T+decode_steps)/1024), unlike standalone low-level exact T. Fixed-g comparisons use the same capacity, layout, bytes and model/data revision.

Native GDN reference fallback observed: True. This does not establish performance with optimized native fast paths or production serving. Rounds share one model process; small speed differences are not independent-process confirmation. FP16 gbest/g1 is an identical-execution control: its observed round spread must be considered before attributing small whole-model effects to grouping. No native logit reference is collected in this speed-only confirmation; finite logits are required and online subgroup equivalence is checked separately in the smoke.

Failures/skips: []
