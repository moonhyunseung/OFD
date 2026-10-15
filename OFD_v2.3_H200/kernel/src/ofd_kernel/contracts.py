from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Variant(str, Enum):
    P = "P"    # plain K, plain V
    K = "K"    # rotated K, plain V
    V = "V"    # plain K, rotated V
    KV = "KV"  # rotated K, rotated V

    @property
    def rotate_k(self) -> bool:
        return self in (Variant.K, Variant.KV)

    @property
    def rotate_v(self) -> bool:
        return self in (Variant.V, Variant.KV)


@dataclass(frozen=True)
class StorageContract:
    head_dim: int = 128
    group_size: int = 32
    page_size: int = 16
    scale_bytes: int = 2

    def __post_init__(self) -> None:
        if self.head_dim <= 0 or self.group_size <= 0 or self.page_size <= 0:
            raise ValueError("head_dim, group_size and page_size must be positive")
        if self.head_dim % self.group_size != 0:
            raise ValueError("head_dim must be divisible by group_size")
        if self.group_size % 2 != 0:
            raise ValueError("group_size must be even for 2-values/byte INT4 packing")
        if self.head_dim & (self.head_dim - 1):
            raise ValueError("head_dim must be a power of two for FWHT")

    @property
    def groups_per_head(self) -> int:
        return self.head_dim // self.group_size

    @property
    def packed_bytes_per_group(self) -> int:
        return self.group_size // 2

    @property
    def packed_data_bytes_per_token_head(self) -> int:
        return self.head_dim // 2

    @property
    def scale_storage_bytes_per_token_head(self) -> int:
        return self.groups_per_head * self.scale_bytes

    @property
    def int4_bytes_per_token_head(self) -> int:
        return (
            self.packed_data_bytes_per_token_head
            + self.scale_storage_bytes_per_token_head
        )

    @property
    def fp16_bytes_per_token_head(self) -> int:
        return self.head_dim * 2

    @property
    def kv_compression_vs_fp16(self) -> float:
        fp = 2 * self.fp16_bytes_per_token_head
        q = 2 * self.int4_bytes_per_token_head
        return fp / q

    def describe(self) -> dict[str, int | float]:
        return {
            "head_dim": self.head_dim,
            "group_size": self.group_size,
            "page_size": self.page_size,
            "groups_per_head": self.groups_per_head,
            "packed_bytes_per_group": self.packed_bytes_per_group,
            "int4_bytes_per_token_head_per_K_or_V": self.int4_bytes_per_token_head,
            "fp16_bytes_per_token_head_per_K_or_V": self.fp16_bytes_per_token_head,
            "kv_compression_vs_fp16": self.kv_compression_vs_fp16,
        }
