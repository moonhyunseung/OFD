from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class TritonTuneCandidate:
    split_size: int
    tokens_per_block: int
    num_warps: int
    num_stages: int

BLACKWELL_CANDIDATES = (
    TritonTuneCandidate(512, 16, 4, 2),
    TritonTuneCandidate(1024, 16, 4, 2),
    TritonTuneCandidate(1024, 32, 4, 3),
    TritonTuneCandidate(2048, 32, 8, 3),
)
