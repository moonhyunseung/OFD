# Order-Flip Decoding for Rotated Quantized KV Caches

A research prototype for a **TurboQuant-compatible decode-path optimization**.

## Overview

This repository explores a simple question:

> If keys and values are stored in a rotated quantized domain, do we really need to inverse-rotate the entire KV cache at every decode step?

The prototype answer is: **not necessarily**.

Instead of reconstructing the whole cache in the original domain, this prototype:
- rotates the query once,
- computes attention scores in the rotated domain,
- aggregates rotated-domain values,
- inverse-rotates only the final output.

## What this is

This repository implements:
- rotated KV cache using a normalized Walsh-Hadamard transform,
- packed int4 groupwise quantization,
- a **naive decode path**,
- an **order-flip decode path**,
- geometry-level scaling experiments for 124M-, 1B-, and 7B-class settings,
- long-context multi-step decode benchmarks.

## Benchmark Results

### Setup
- **Device:** CUDA
- **Dtype:** bfloat16
- **Geometry:** 7B-class
- **Heads:** 32
- **Hidden size:** 4096
- **Head dim:** 128
- **Cache length:** 8192
- **Decode steps:** 128
- **Batch size:** 1

### Single-step correctness
- FP vs naive-TQ rel error: `1.019021e-01`
- FP vs order-flip-TQ rel error: `1.019342e-01`
- naive-TQ vs order-flip-TQ rel error: `3.857114e-03` ✅

### Multi-step decode correctness
- FP vs naive-TQ rel error: `1.006369e-01`
- FP vs order-flip-TQ rel error: `1.006517e-01`
- naive-TQ vs order-flip-TQ rel error: `3.789389e-03` ✅

### Multi-step decode performance
- FP baseline total: `340.862 ms`
- TQ naive total: `18129.655 ms`
- TQ order-flip total: `4105.882 ms`

**Throughput:**
- FP baseline: `2.6630 ms/token`
- TQ naive: `141.6379 ms/token`
- TQ order-flip: `32.0772 ms/token`
- **Speedup (naive / order-flip): `4.42x`** ⭐

### Peak VRAM
- FP baseline: `1.42 GB`
- TQ naive: `1.96 GB`
- TQ order-flip: `1.90 GB`
- Peak ratio: `1.03x` (negligible overhead)

### KV cache storage
- FP KV bytes: `128.00 MB`
- packed-int4 rotated KV bytes: `36.00 MB`
- **Compression vs FP: `3.56x`** ⭐
- Effective bits/value: `4.50`

## Decode Paths

### 1. Naive Path
At each decode step:
1. dequantize the rotated cache
2. inverse-rotate the whole cache
3. append the new token
4. compute attention in the original domain
5. rotate and repack the updated cache

### 2. Order-Flip Path
At each decode step:
1. dequantize the rotated cache
2. rotate the query
3. append the new token in the rotated domain
4. compute attention in the rotated domain
5. inverse-rotate only the final output
6. repack the updated rotated cache

## Key Findings

✅ **4.42× speedup** within the quantization framework  
✅ **3.56× cache compression** with negligible accuracy loss  
✅ **0.38% accuracy difference** between naive and order-flip  
✅ **Scales from 124M to 7B** model sizes  

## Limitations

- This is a prototype, not a full TurboQuant reimplementation
- 12× slower than FP32 baseline (quantization overhead)
- No custom CUDA kernels (pure PyTorch)
- No 1-bit QJL residual correction
- No production serving stack

## Repository Structure