import math
import unittest

import torch

from ofd_kernel.contracts import StorageContract, Variant
from ofd_kernel.reference.fwht import fwht_normalized
from ofd_kernel.reference.paged import materialize_logical_cache, pack_dense_to_paged_int4
from ofd_kernel.runtime.policy import StaticPolicy
from ofd_kernel.reference.quant import (
    dequantize_int4_packed_groupwise,
    quantize_int4_packed_groupwise,
)
from ofd_kernel.reference.attention import (
    attention_decode,
    paged_quantized_variant_attention,
    quantized_variant_attention,
    split_kv_attention,
    split_kv_partials,
)
from ofd_kernel.runtime.launcher import next_power_of_2, num_splits_for


class KernelV2CPUTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1337)

    def test_storage_contract_matches_ofd(self):
        c = StorageContract(head_dim=128, group_size=32, page_size=16)
        self.assertEqual(c.groups_per_head, 4)
        self.assertEqual(c.packed_bytes_per_group, 16)
        self.assertEqual(c.int4_bytes_per_token_head, 72)
        self.assertAlmostEqual(c.kv_compression_vs_fp16, 512 / 144, places=10)

    def test_pack_encoding_contract(self):
        x = torch.tensor(
            [[-8.0, -7.0, -1.0, 0.0, 1.0, 6.0, 7.0, 8.0]],
            dtype=torch.float32,
        )
        q = quantize_int4_packed_groupwise(x, group_size=8)
        self.assertEqual(q.packed.dtype, torch.uint8)
        self.assertEqual(q.scale.dtype, torch.float16)
        self.assertEqual(q.packed.shape[-1], 4)
        y = dequantize_int4_packed_groupwise(q)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        # absmax/7 is symmetric, so ordinary quantization should decode within
        # the representable range and never require an out-of-contract code.
        self.assertTrue(torch.isfinite(y).all())

    def test_fwht_self_inverse(self):
        x = torch.randn(3, 5, 128)
        y = fwht_normalized(fwht_normalized(x))
        self.assertTrue(torch.allclose(x, y, atol=2e-5, rtol=2e-5))

    def test_unquantized_basis_invariance(self):
        b, h, t, d = 2, 4, 33, 64
        q = torch.randn(b, h, d)
        k = torch.randn(b, h, t, d)
        v = torch.randn(b, h, t, d)

        ref = attention_decode(q, k, v)
        qr = fwht_normalized(q)
        kr = fwht_normalized(k)
        vr = fwht_normalized(v)

        k_only = attention_decode(qr, kr, v)
        v_only = fwht_normalized(attention_decode(q, k, vr))
        full = fwht_normalized(attention_decode(qr, kr, vr))

        for out in (k_only, v_only, full):
            self.assertTrue(torch.allclose(ref, out, atol=5e-5, rtol=5e-5))

    def test_split_kv_matches_direct_gqa(self):
        b, hq, hkv, t, d = 2, 8, 2, 137, 64
        q = torch.randn(b, hq, d)
        k = torch.randn(b, hkv, t, d)
        v = torch.randn(b, hkv, t, d)

        ref = attention_decode(q, k, v)
        for split in (16, 31, 64, 128):
            got = split_kv_attention(q, k, v, split)
            self.assertTrue(torch.allclose(ref, got, atol=5e-5, rtol=5e-5))

    def test_ragged_split_with_empty_splits_is_finite_and_correct(self):
        b, hq, hkv, t, d = 3, 8, 2, 129, 64
        q = torch.randn(b, hq, d)
        k = torch.randn(b, hkv, t, d)
        v = torch.randn(b, hkv, t, d)
        seq_lens = torch.tensor([5, 37, 129])

        ref = attention_decode(q, k, v, seq_lens=seq_lens)
        # 6 splits is intentionally larger than needed for the first two reqs.
        got = split_kv_attention(
            q,
            k,
            v,
            split_size=32,
            seq_lens=seq_lens,
            num_splits=6,
        )
        self.assertTrue(torch.isfinite(got).all())
        self.assertTrue(torch.allclose(ref, got, atol=5e-5, rtol=5e-5))

        state = split_kv_partials(
            q,
            k,
            v,
            split_size=32,
            seq_lens=seq_lens,
            num_splits=6,
        )
        # request 0 only has tokens in split 0; remaining states are neutral.
        self.assertTrue(torch.isneginf(state.m[1:, 0]).all())
        self.assertTrue(torch.equal(state.l[1:, 0], torch.zeros_like(state.l[1:, 0])))
        self.assertTrue(torch.equal(state.o[1:, 0], torch.zeros_like(state.o[1:, 0])))

    def test_quantized_split_and_direct_match(self):
        b, h, t, d = 1, 4, 129, 64
        q = torch.randn(b, h, d)
        k = torch.randn(b, h, t, d)
        v = torch.randn(b, h, t, d)

        for variant in Variant:
            a = quantized_variant_attention(q, k, v, variant, group_size=32)
            b_out = quantized_variant_attention(
                q, k, v, variant, group_size=32, split_size=37
            )
            self.assertTrue(torch.allclose(a, b_out, atol=5e-5, rtol=5e-5))

    def test_randomized_physical_pages_roundtrip(self):
        b, hkv, t, d = 3, 2, 41, 64
        seq_lens = torch.tensor([7, 23, 41])
        k = torch.randn(b, hkv, t, d)
        v = torch.randn(b, hkv, t, d)

        cache = pack_dense_to_paged_int4(
            k,
            v,
            seq_lens,
            page_size=8,
            group_size=32,
            seed=20261012,
        )
        # Random placement should not collapse to the identity except by a
        # vanishingly unlikely coincidence for this deterministic seed.
        used = cache.block_table[cache.block_table >= 0]
        self.assertFalse(torch.equal(used, torch.arange(used.numel())))

        k_logical, v_logical = materialize_logical_cache(cache)
        # Compare only valid tokens against dense QDQ.
        k_qdq = dequantize_int4_packed_groupwise(
            quantize_int4_packed_groupwise(k, 32)
        )
        v_qdq = dequantize_int4_packed_groupwise(
            quantize_int4_packed_groupwise(v, 32)
        )
        for bi, n in enumerate(seq_lens.tolist()):
            self.assertTrue(torch.equal(k_logical[bi, :, :n], k_qdq[bi, :, :n]))
            self.assertTrue(torch.equal(v_logical[bi, :, :n], v_qdq[bi, :, :n]))

    def test_paged_oracle_matches_dense_all_variants_ragged(self):
        b, hq, hkv, t, d = 3, 8, 2, 65, 64
        seq_lens = torch.tensor([9, 38, 65])
        q = torch.randn(b, hq, d)
        k = torch.randn(b, hkv, t, d)
        v = torch.randn(b, hkv, t, d)

        for variant in Variant:
            dense = quantized_variant_attention(
                q,
                k,
                v,
                variant,
                group_size=32,
                split_size=16,
                seq_lens=seq_lens,
                num_splits=6,
            )
            paged, _ = paged_quantized_variant_attention(
                q,
                k,
                v,
                variant,
                seq_lens=seq_lens,
                group_size=32,
                page_size=8,
                split_size=16,
                num_splits=6,
                seed=4242,
            )
            self.assertTrue(torch.allclose(dense, paged, atol=5e-5, rtol=5e-5))

    def test_dual_cutoff_policy(self):
        p = StaticPolicy.dual_cutoff(n_layers=24, k_cut=2, v_cut=22)
        self.assertEqual(p[0].variant, Variant.KV)
        self.assertEqual(p[1].variant, Variant.KV)
        self.assertEqual(p[2].variant, Variant.V)
        self.assertEqual(p[21].variant, Variant.V)
        self.assertEqual(p[22].variant, Variant.P)
        self.assertEqual(p[23].variant, Variant.P)

    def test_runtime_shape_helpers(self):
        self.assertEqual(next_power_of_2(1), 1)
        self.assertEqual(next_power_of_2(5), 8)
        self.assertEqual(num_splits_for(torch.tensor([5, 37, 129]), 32), 5)


if __name__ == "__main__":
    unittest.main()
