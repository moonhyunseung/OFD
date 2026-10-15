# H200 screening results — 2026-10-12

Archived raw CSVs, logs, and manifests from the H200 workspace. Both E00–E02 runs are retained.

- E00 CPU reference, E01 GPU compile, E02 numerical correctness: completed, PASS.
- E04a core attention: completed, PASS under its frozen smoke gate.
- E07 long-context core attention: completed, observational (NOT_APPLICABLE gate).

E04a and E07 exclude Q/output transforms from timing. These are core-attention measurements, not actual-model end-to-end results. E04b/E05/E09 results are not present in this archive.

The raw manifests retain original commit IDs and environment metadata. The source snapshot recorded as a621fc4613a731f9171b64ef7739b94392f16b23 has tree SHA 228a318b9461fc59420fd79451254fa388d9f507, matching the 70-file source snapshot imported into this branch.
