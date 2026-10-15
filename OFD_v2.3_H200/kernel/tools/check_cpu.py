from __future__ import annotations

import json
import torch

from ofd_kernel.contracts import StorageContract, Variant
from ofd_kernel.reference.attention import (
    paged_quantized_variant_attention,
    quantized_variant_attention,
)


def rel(a, b):
    return ((a.float() - b.float()).norm() / (a.float().norm() + 1e-12)).item()


def main():
    torch.manual_seed(1337)

    contract = StorageContract()
    print("STORAGE CONTRACT")
    print(json.dumps(contract.describe(), indent=2))

    b, hq, hkv, t, d = 3, 8, 2, 257, 128
    q = torch.randn(b, hq, d)
    k = torch.randn(b, hkv, t, d)
    v = torch.randn(b, hkv, t, d)
    seq_lens = torch.tensor([17, 93, 257])

    print("\nRAGGED + PAGED + RANDOM-PHYSICAL-PAGE SELF-CHECK")
    worst = 0.0
    for variant in Variant:
        dense = quantized_variant_attention(
            q,
            k,
            v,
            variant,
            group_size=32,
            split_size=64,
            seq_lens=seq_lens,
            num_splits=6,
        )
        paged, cache = paged_quantized_variant_attention(
            q,
            k,
            v,
            variant,
            seq_lens=seq_lens,
            group_size=32,
            page_size=16,
            split_size=64,
            num_splits=6,
            seed=20261012,
        )
        e = rel(dense, paged)
        worst = max(worst, e)
        print(
            f"{variant.value:>2}: paged/dense rel={e:.3e} | "
            f"physical_pages={cache.num_pages}"
        )

    if not worst < 1e-5:
        raise RuntimeError(f"CPU paged oracle mismatch: worst={worst}")

    print("\nCPU_REFERENCE_OK")


if __name__ == "__main__":
    main()
