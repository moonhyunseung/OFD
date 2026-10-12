# Prior-art boundary — working notes

This package does **not** claim orthogonal rotation cancellation inside attention or rotated-domain KV quantization as new mathematical primitives.

Relevant design families include QuaRot, TurboQuant, RotateKV, layer-wise mixed KV quantization methods, and modern paged/split attention implementations such as FlashInfer/FlashAttention-derived systems.

The research question here is narrower: how to implement and measure direct packed-INT4 paged decode with static P/K/V/KV representation choices, and whether that representation flexibility can be retained at approximately Plain-INT4 decode cost.

Before publication or patent work, re-run a formal prior-art review and replace these working notes with verified bibliographic citations.
