# Storage contract

This file is a freeze boundary. Changing it creates a new experiment lineage.

## Logical tensors

- Query: `[B, Hq, D]`
- Packed K/V: `[physical_page, Hkv, PAGE, group, bytes_per_group]`
- K/V scales: `[physical_page, Hkv, PAGE, group]`
- Block table: `[B, max_logical_pages]`
- Sequence lengths: `[B]`

## Frozen defaults

- `D`: 64 or 128 in screening; power of two for FWHT
- `GROUP_SIZE = 32`
- `PAGE_SIZE = 16`
- packed bytes/group = 16
- scale dtype = FP16
- encoding = signed 4-bit represented as `q+8` in uint4
- low nibble stores the first value, high nibble the second
- quantizer scale = `absmax / 7`

4-bit storage can represent `[-8,7]`; the symmetric absmax/7 quantizer normally occupies `[-7,7]`.

For D=128, each K or V token/head uses 64 packed-data bytes + 8 scale bytes = 72 bytes. K+V therefore uses 144 bytes versus 512 bytes in FP16, i.e. 3.555...x representation compression.
