# OFD — Order-Flip Decoding for Rotated Quantized KV Caches

**OFD is a systems/reproduction research prototype for studying decode execution over rotated, quantized KV caches.**

The repository focuses on a practical implementation question:

> When a KV cache is already stored in an orthogonally rotated quantized domain, can attention consume that representation directly instead of reconstructing the full cache in the original domain at every decode step?

The answer is algebraically yes for an orthogonal rotation, and this repository measures the implementation consequences of that execution order.

> **Scope / novelty note**
>
> Orthogonal-rotation cancellation inside attention and rotated-domain KV quantization are not claimed here as new mathematical primitives. Closely related ideas appear in prior rotation-based quantization work such as QuaRot, TurboQuant, and RotateKV. OFD is maintained as an independent systems prototype and experimental testbed for the decode-path formulation, packed-INT4 behavior, long-context scaling, and serving-oriented follow-up work.

---

## Core identity

Let `R` be an orthogonal rotation (`R R^T = I`) and use row-vector notation:

```text
q_R = q R
K_R = K R
V_R = V R
```

Then attention scores are unchanged:

```text
q_R K_R^T
= q R (K R)^T
= q R R^T K^T
= q K^T
```

If `p = softmax(q K^T)`, value aggregation can also stay in the rotated domain:

```text
y_R = p V_R = p V R
y   = y_R R^T
```

So a decode implementation does not need to inverse-rotate every cached key/value merely to perform attention. It can rotate the query, consume the rotated cache, aggregate rotated values, and transform only the final attention output.

Quantization makes the two numerical paths only approximately equivalent because quantization does not generally commute with rotation. The experiments in this repository measure that discrepancy rather than assuming exact equivalence after quantization.

---

## What this repository is

This public snapshot contains a compact research implementation of:

- normalized Walsh–Hadamard rotation,
- signed groupwise packed INT4 KV quantization,
- a deliberately naive rotated-KV decode path,
- an order-flipped rotated-domain decode path,
- single-step and multi-step numerical comparisons,
- synthetic model-geometry scaling experiments,
- long-context decode measurements.

The goal is **mechanism isolation**, not production benchmarking. The code is intentionally small enough to inspect and modify.

### What this repository is not

- not a claim that rotated-domain attention itself is a new primitive,
- not a full TurboQuant, QuaRot, or RotateKV reimplementation,
- not an optimized FlashAttention/PagedAttention serving backend,
- not evidence that this PyTorch prototype is faster than optimized FP attention,
- not an end-task quality benchmark,
- not a production-ready inference engine.

Ongoing research extensions are intentionally not all included in this public snapshot.

---

## Decode paths under test

### 1. Naive reconstruction path

At each decode step, the prototype conceptually:

1. dequantizes the rotated cache,
2. inverse-rotates cached K/V back to the original domain,
3. performs attention in the original domain,
4. updates the cache,
5. rotates/re-quantizes as required by the prototype.

This path is useful as a deliberately expensive control: it exposes the cost of repeatedly materializing the cache in the original basis.

### 2. Order-flipped path

At each decode step:

1. keep K/V in the rotated quantized representation,
2. rotate the query once,
3. compute attention scores against rotated keys,
4. aggregate rotated values,
5. inverse-rotate only the final attention output.

The central systems question is therefore not whether the algebra exists, but **how much unnecessary cache-wide transformation/materialization can be removed from a rotated-quantized decode implementation**.

---

## Historical prototype benchmark

The numbers below are preserved as the original public prototype measurement. They should be interpreted as a comparison **inside this implementation**, not as a production serving result.

### Setup

- **Device:** CUDA
- **Dtype:** bfloat16
- **Synthetic geometry:** 7B-class
- **Heads:** 32
- **Hidden size:** 4096
- **Head dimension:** 128
- **Cache length:** 8192
- **Decode steps:** 128
- **Batch size:** 1

### Numerical comparison

Single step:

- FP vs naive rotated-INT4 relative error: `1.019021e-01`
- FP vs order-flip rotated-INT4 relative error: `1.019342e-01`
- naive vs order-flip relative-output difference: `3.857114e-03`

Multi-step:

- FP vs naive rotated-INT4 relative error: `1.006369e-01`
- FP vs order-flip rotated-INT4 relative error: `1.006517e-01`
- naive vs order-flip relative-output difference: `3.789389e-03`

These are tensor-level numerical measurements, **not task accuracy percentages**.

### Decode time inside the prototype

- FP baseline total: `340.862 ms`
- naive rotated-quantized path: `18129.655 ms`
- order-flip rotated-quantized path: `4105.882 ms`

Per decode token:

- FP baseline: `2.6630 ms/token`
- naive rotated-quantized path: `141.6379 ms/token`
- order-flip path: `32.0772 ms/token`
- **naive / order-flip: `4.42×`**

The `4.42×` number is specifically the reduction relative to the deliberately naive rotated-quantized implementation. It is **not** a claim of a `4.42×` speedup over optimized FP/BF16 attention.

### KV representation size

- FP KV: `128.00 MB`
- packed INT4 rotated KV: `36.00 MB`
- compression relative to FP: **`3.56×`**
- effective storage: `4.50 bits/value`

This storage ratio follows from the packed INT4 representation plus group scales used by the prototype.

---

## What the current evidence supports

The public experiment supports three narrow conclusions:

1. **Rotated-domain execution is numerically close to the corresponding naive rotated-quantized path** in this prototype.
2. **Avoiding repeated cache-wide basis reconstruction removes a large amount of work from that naive implementation.**
3. **Packed groupwise INT4 substantially reduces KV representation size.**

It does **not** by itself establish production throughput gains, downstream model quality, or superiority over optimized quantized-serving systems.

---

## Relation to prior work

OFD sits in an existing line of work on orthogonal rotations and low-bit LLM inference. Relevant examples include:

- **QuaRot** — rotation-based outlier removal / quantization for LLM inference  
  <https://arxiv.org/abs/2404.00456>
- **TurboQuant** — rotation-based KV-cache quantization and efficient decoding  
  <https://arxiv.org/search/?query=TurboQuant&searchtype=all>
- **RotateKV** — rotation-based ultra-low-bit KV-cache quantization  
  <https://arxiv.org/abs/2501.16383>

These references are listed to make the boundary of this repository explicit. The public OFD code should be read as a **systems exploration and reproducible prototype within this broader design family**, not as an assertion that rotation cancellation in attention originated here.

---

## Research direction

The practical target is long-context inference where KV-cache capacity and memory traffic become important serving constraints. Useful evaluation axes include:

- KV bytes per token,
- maximum resident context / sessions per GPU,
- same-memory-budget throughput,
- TTFT and TPOT,
- attention/output error,
- downstream perplexity or task quality,
- behavior under paged KV allocation and real serving workloads.

This repository remains intentionally semi-open: the baseline mechanism and reproducible prototype are public, while not every ongoing experimental extension is published immediately.

---

## Repository structure

```text
OFD/
├── README.md
└── experiments/
    └── benchmark_packed_int4_multistep.py
```

The public `main` branch represents the baseline research snapshot.

---

## Status

**Research prototype / experimental code.**

Results should be reproduced on the target hardware and serving stack before being used for deployment or cost projections.
