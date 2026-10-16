# Prior-art boundary — working bibliography and claim limits

This file is a **working prior-art map**, not a completed novelty search.

OFD must not claim the following as new primitives by themselves:

- orthogonal/rotation cancellation in attention;
- Hadamard/randomized-Hadamard preprocessing for quantization;
- low-bit KV-cache quantization;
- paged/split attention;
- GQA reuse;
- generic INT4 Tensor-Core use;
- moving a linear transform into adjacent weights when algebraically legal.

The prospective novelty, if any, must come from the specific combination of operator quotient/equivalence analysis, representation-native packed contraction, circuit-aligned factorization, and demonstrated end-to-end benefit.

## 1. Exact/IO-aware attention

### FlashAttention

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.** arXiv:2205.14135 (2022).

Boundary for OFD:

- exact attention can be redesigned from an IO-complexity model rather than by preserving a naive materialization order;
- OFD must therefore justify its dataflow/work reduction analytically, not just by timing a custom kernel.

## 2. Rotation / incoherence quantization

### QuIP#

Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, Christopher De Sa. **QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks.** arXiv:2402.04396 (2024).

Boundary for OFD:

- randomized Hadamard incoherence processing is prior art;
- Hadamard flattening/coherence reduction cannot be claimed as an OFD novelty;
- OFD may still study a different question: whether basis geometry reduces **attention circuit work** or aligns visible subspaces with hardware blocks.

### QuaRot

Saleh Ashkboos et al. **QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs.** arXiv:2404.00456 (2024).

Boundary for OFD:

- exact rotation invariances used to enable low-bit inference, including KV-related rotations, are prior art;
- OFD's order-flip/rotation cancellation is an enabling algebraic observation, not a sufficient novelty claim.

### Provable Quantization with Randomized Hadamard Transform

Ying Feng, Piotr Indyk, Michael Kapralov, Dmitry Krachun, Boris Prokhorov. **Provable Quantization with Randomized Hadamard Transform.** arXiv:2605.13810 (2026).

Boundary for OFD:

- theoretical guarantees for randomized-Hadamard-based quantization now exist;
- any OFD claim about high-dimensional flattening/concentration must be stated consistently with this literature and must distinguish quantizer error from attention-operator-visible error.

## 3. KV rotation / RoPE placement

### RotateKV

Zunhai Su, Zhe Chen, Wang Shen, Hanyu Wei, Linge Li, Huangqi Yu, Kehong Yuan. **RotateKV: Accurate and Robust 2-Bit KV Cache Quantization for LLMs via Outlier-Aware Adaptive Rotations.** arXiv:2501.16383 (2025).

Boundary for OFD:

- low-bit KV quantization with rotation and explicit treatment of rotation placement relative to RoPE is prior art;
- OFD's RoPE/basis-placement analysis must be compared against RotateKV rather than presented as an isolated discovery.

## 4. Low-bit serving / dequantization overhead

### QServe

Yujun Lin, Haotian Tang, Shang Yang, Zhekai Zhang, Guangxuan Xiao, Chuang Gan, Song Han. **QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving.** arXiv:2405.04532 (2024).

Boundary for OFD:

- runtime dequantization/conversion overhead in low-bit serving is already a known systems problem;
- co-designing quantization with execution is prior art;
- OFD must not claim merely that dequantization is expensive or that low-bit storage requires representation-aware kernels;
- a stronger OFD contribution would need a distinct operator factorization / quotient-space reduction / circuit mapping and evidence that it removes work beyond existing dequant-aware designs.

## 5. Transform coding / subspace compression

### KV Cache Transform Coding

Konrad Staniszewski, Adrian Łańcucki. **KV Cache Transform Coding for Compact Storage in LLM Inference.** arXiv:2511.01815 (2025).

Boundary for OFD:

- PCA/subspace transform coding of KV states exists;
- an OFD feature-subspace claim must distinguish query-conditioned operator nullity / hardware-block alignment from generic data-compression PCA.

## 6. NVIDIA hardware / compiler boundary

### PTX ISA

NVIDIA **Parallel Thread Execution ISA** documentation (current PTX family; verify exact version archived with a claim).

Relevant fact for the current H200 target:

- PTX documents `.u4/.s4` MMA families with reduction dimension `K=32` (for example `m8n8k32` / `m16n8k32` families, with architecture requirements below SM90 and therefore available on Hopper-class targets).

Boundary:

- this proves an available ISA family, not that Triton or a particular source expression emits it;
- every OFD instruction-level claim must archive generated PTX/SASS.

### CUTLASS Hopper documentation

NVIDIA CUTLASS documentation describes Hopper mixed-input kernel generation including narrow integer (`int4/uint4`) with `f16/bf16` combinations and conversion/scale/zero-point modes.

Boundary:

- library support is evidence of an available implementation family, not proof of one particular native mixed-dtype MMA instruction;
- OFD must inspect generated lowering and tile utilization for the exact `(r x 32) @ (32 x N)` geometry.

## 7. Current OFD novelty hypothesis — not a claim

The strongest current hypothesis is not “rotation + INT4” and not “grouping helps INT4.” It is:

> **Attention's operator equivalence classes/null spaces and compressed-code algebra may permit an exact or controlled-approximate factorization whose natural 32-wide contractions map to low-bit GPU primitives without constructing dequantized KV working matrices, and whose useful computational dimension can be smaller than the stored feature/token dimensions.**

This still requires:

1. E14 evidence that the theoretically available quotient/subspace structure is stable/exploitable in real model states;
2. a formal comparison against prior low-bit attention/KV kernels;
3. E15 generated-code evidence that the proposed factorization actually reduces circuit work;
4. complete-model and same-memory-budget validation.

Do not promote the hypothesis to a novelty statement before a dedicated literature review covering 2025–2026 low-bit attention, KV quantization, mixed-dtype kernels, transform coding, sketching/subspace methods and architecture-specific inference systems.
