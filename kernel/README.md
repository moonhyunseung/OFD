# OFD Kernel Workspace — Triton v2.3

A private/semi-open kernel-development workspace separated into code, documents, experiment protocols, and generated results.

## Status

- CPU/reference contract: ready and tested
- paged/ragged/random-physical-page CPU oracle: ready
- Triton two-stage split-KV kernel: correctness-first draft, not yet compiled on the target Blackwell host
- Blackwell optimization: not started
- grouped-GQA reuse: not implemented
- production/serving claims: not authorized by current gates

## Layout

```text
docs/          design contracts and claim boundaries
src/           reference, Triton, runtime code
tests/         CPU/GPU validation
experiments/   frozen protocols E00–E09
results/       generated run artifacts (gitignored)
tools/         environment, runner, benchmark, summarizer
```

## GPU-free validation

```bash
export PYTHONPATH="$PWD/kernel/src"
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s kernel/tests/cpu -p 'test_*.py' -v
CUDA_VISIBLE_DEVICES="" python kernel/tools/check_cpu.py
python kernel/tools/run_experiment.py --all --dry-run
```

## Target-host execution order

```bash
python kernel/experiments/E00_cpu_reference/run.py
python kernel/experiments/E01_gpu_compile/run.py
python kernel/experiments/E02_correctness/run.py
```

Do not skip gates. See `experiments/EXPERIMENT_PLAN.md`, `docs/CLAIMS_AND_GATES.md`, and `docs/MEASUREMENT_CONTRACT.md`.
