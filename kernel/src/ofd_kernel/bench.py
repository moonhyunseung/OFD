from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class BenchmarkCase:
    context: int
    batch: int
    q_per_kv: int
    head_dim: int = 128
    page_size: int = 16
    group_size: int = 32

CONTEXTS = (2048, 4096, 8192, 16384, 32768, 65536, 131072)
BATCHES = (1, 2, 4, 8, 16, 32)
Q_PER_KV = (1, 4, 8)

def full_matrix():
    return [BenchmarkCase(t,b,q) for t in CONTEXTS for b in BATCHES for q in Q_PER_KV]

def smoke_matrix():
    return [
        BenchmarkCase(8192,1,1), BenchmarkCase(8192,4,4),
        BenchmarkCase(32768,1,1), BenchmarkCase(32768,4,4),
    ]

def long_context_matrix():
    return [BenchmarkCase(t,b,q) for t in (8192,16384,32768,65536,131072) for b in (1,4,8) for q in (1,4)]

def same_memory_batch(fp_batch: int, compression: float = 3.5555555556) -> int:
    return max(1, int(fp_batch * compression))
