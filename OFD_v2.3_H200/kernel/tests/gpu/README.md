# GPU tests

These tests are intentionally not executed by the installer. Run them only on the target CUDA host after E00 passes.

- E01: compile smoke, no performance claim.
- E02: numerical correctness vs CPU paged oracle.
- Later experiments remain gated by E02/E04 and their implementation status.
