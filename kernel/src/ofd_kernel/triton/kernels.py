"""Triton v2 kernel draft.

This file is intentionally import-safe and GPU-execution-free until the first
Blackwell/V100 compile pass. The kernel architecture is already fixed:

    grid = (request, q_head, split)
    direct packed INT4 load -> register unpack/dequant
    -> local online softmax state (m, l, o)
    -> partial state store
    -> second-stage merge

Important representation rule:
The attention kernel itself does not need P/K/V/KV branches.
It consumes whichever K/V basis the layer stores:
- K-rotation => caller supplies q in the same rotated basis.
- V-rotation => kernel returns the weighted sum in V's stored basis;
  caller applies one output FWHT after the split merge.

That keeps the hot attention kernel branch-free.
"""

from __future__ import annotations

try:
    import triton
    import triton.language as tl
except Exception:  # Allows CPU-only static/unit tests.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def splitkv_int4_partials_kernel(
        q_ptr,
        k_packed_ptr,
        k_scale_ptr,
        v_packed_ptr,
        v_scale_ptr,
        block_table_ptr,
        seq_lens_ptr,
        partial_m_ptr,
        partial_l_ptr,
        partial_o_ptr,

        stride_q_b: tl.constexpr,
        stride_q_h: tl.constexpr,

        stride_k_page: tl.constexpr,
        stride_k_head: tl.constexpr,
        stride_k_tok: tl.constexpr,
        stride_k_group: tl.constexpr,

        stride_ks_page: tl.constexpr,
        stride_ks_head: tl.constexpr,
        stride_ks_tok: tl.constexpr,
        stride_ks_group: tl.constexpr,

        stride_v_page: tl.constexpr,
        stride_v_head: tl.constexpr,
        stride_v_tok: tl.constexpr,
        stride_v_group: tl.constexpr,

        stride_vs_page: tl.constexpr,
        stride_vs_head: tl.constexpr,
        stride_vs_tok: tl.constexpr,
        stride_vs_group: tl.constexpr,

        stride_bt_b: tl.constexpr,
        stride_bt_p: tl.constexpr,

        stride_pm_b: tl.constexpr,
        stride_pm_h: tl.constexpr,
        stride_pm_s: tl.constexpr,

        stride_po_b: tl.constexpr,
        stride_po_h: tl.constexpr,
        stride_po_s: tl.constexpr,

        NUM_Q_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        SPLIT_SIZE: tl.constexpr,
        TOKENS_PER_BLOCK: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
    ):
        """Stage-1 split-KV decode kernel.

        Contract:
        q                  [B, Hq, D]
        packed K/V         [num_pages, Hkv, PAGE_SIZE, G, GROUP_SIZE/2]
        scales K/V         [num_pages, Hkv, PAGE_SIZE, G]
        block_table        [B, max_pages]
        partial_m/l        [B, Hq, NUM_SPLITS]
        partial_o          [B, Hq, NUM_SPLITS, D]

        Current draft assumes Hq % Hkv == 0 and maps q-head to the
        corresponding shared KV head. The first GPU pass should compile and
        validate this baseline before adding Blackwell-specific scheduling.
        """
        b = tl.program_id(0)
        qh = tl.program_id(1)
        split = tl.program_id(2)

        q_per_kv = NUM_Q_HEADS // NUM_KV_HEADS
        kvh = qh // q_per_kv

        d = tl.arange(0, HEAD_DIM)
        q = tl.load(q_ptr + b * stride_q_b + qh * stride_q_h + d).to(tl.float32)

        group = d // GROUP_SIZE
        within_group = d % GROUP_SIZE
        byte_in_group = within_group // 2
        hi_nibble = (within_group & 1) != 0

        seq_len = tl.load(seq_lens_ptr + b)
        split_start = split * SPLIT_SIZE
        split_end = tl.minimum(split_start + SPLIT_SIZE, seq_len)
        split_active = split_start < seq_len

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

        # Static loop count keeps codegen predictable. Mask handles the tail.
        n_blocks: tl.constexpr = (SPLIT_SIZE + TOKENS_PER_BLOCK - 1) // TOKENS_PER_BLOCK

        for block_idx in range(0, n_blocks):
            block_start = split_start + block_idx * TOKENS_PER_BLOCK
            tok = block_start + tl.arange(0, TOKENS_PER_BLOCK)
            tok_mask = tok < split_end
            block_active = block_start < split_end

            logical_page = tok // PAGE_SIZE
            page_off = tok % PAGE_SIZE
            phys_page = tl.load(
                block_table_ptr + b * stride_bt_b + logical_page * stride_bt_p,
                mask=tok_mask,
                other=0,
            )

            # [TILE_T, D]. Each packed byte is intentionally loaded twice for
            # the two nibbles in this correctness-first draft. The first GPU
            # optimization pass should vectorize one byte -> two values.
            k_byte_ptrs = (
                k_packed_ptr
                + phys_page[:, None] * stride_k_page
                + kvh * stride_k_head
                + page_off[:, None] * stride_k_tok
                + group[None, :] * stride_k_group
                + byte_in_group[None, :]
            )
            k_code = tl.load(
                k_byte_ptrs,
                mask=tok_mask[:, None],
                other=0,
            ).to(tl.int32)
            k_lo = k_code & 0x0F
            k_hi = (k_code >> 4) & 0x0F
            k_q = tl.where(hi_nibble[None, :], k_hi, k_lo) - 8

            k_scale_ptrs = (
                k_scale_ptr
                + phys_page[:, None] * stride_ks_page
                + kvh * stride_ks_head
                + page_off[:, None] * stride_ks_tok
                + group[None, :] * stride_ks_group
            )
            ks = tl.load(
                k_scale_ptrs,
                mask=tok_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            k = k_q.to(tl.float32) * ks

            score = tl.sum(k * q[None, :], axis=1) * (1.0 / tl.sqrt(float(HEAD_DIM)))
            score = tl.where(tok_mask, score, -float("inf"))

            block_max_raw = tl.max(score, axis=0)
            block_max = tl.where(block_active, block_max_raw, m_i)
            m_new = tl.maximum(m_i, block_max)

            # Empty/tail blocks must never evaluate an effective -inf-(-inf)
            # state into the accumulated result. Safe operands keep even the
            # non-selected tl.where branch finite for ragged batches.
            old_m_safe = tl.where(block_active, m_i, 0.0)
            new_m_safe = tl.where(block_active, m_new, 0.0)
            alpha_raw = tl.exp(old_m_safe - new_m_safe)
            alpha = tl.where(block_active, alpha_raw, 1.0)

            score_safe = tl.where(tok_mask, score, 0.0)
            p_raw = tl.exp(score_safe - new_m_safe)
            p = tl.where(tok_mask & block_active, p_raw, 0.0)

            v_byte_ptrs = (
                v_packed_ptr
                + phys_page[:, None] * stride_v_page
                + kvh * stride_v_head
                + page_off[:, None] * stride_v_tok
                + group[None, :] * stride_v_group
                + byte_in_group[None, :]
            )
            v_code = tl.load(
                v_byte_ptrs,
                mask=tok_mask[:, None],
                other=0,
            ).to(tl.int32)
            v_lo = v_code & 0x0F
            v_hi = (v_code >> 4) & 0x0F
            v_q = tl.where(hi_nibble[None, :], v_hi, v_lo) - 8

            v_scale_ptrs = (
                v_scale_ptr
                + phys_page[:, None] * stride_vs_page
                + kvh * stride_vs_head
                + page_off[:, None] * stride_vs_tok
                + group[None, :] * stride_vs_group
            )
            vs = tl.load(
                v_scale_ptrs,
                mask=tok_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            v = v_q.to(tl.float32) * vs

            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = tl.where(block_active, m_new, m_i)

        # Explicit neutral state for a split beyond this request's seq_len.
        m_store = tl.where(split_active, m_i, -float("inf"))
        l_store = tl.where(split_active, l_i, 0.0)
        acc_store = tl.where(split_active, acc, 0.0)

        pm_base = b * stride_pm_b + qh * stride_pm_h + split * stride_pm_s
        tl.store(partial_m_ptr + pm_base, m_store)
        tl.store(partial_l_ptr + pm_base, l_store)

        po_base = b * stride_po_b + qh * stride_po_h + split * stride_po_s
        tl.store(partial_o_ptr + po_base + d, acc_store)


    @triton.jit
    def merge_splitkv_kernel(
        partial_m_ptr,
        partial_l_ptr,
        partial_o_ptr,
        out_ptr,

        stride_pm_b: tl.constexpr,
        stride_pm_h: tl.constexpr,
        stride_pm_s: tl.constexpr,

        stride_po_b: tl.constexpr,
        stride_po_h: tl.constexpr,
        stride_po_s: tl.constexpr,

        stride_o_b: tl.constexpr,
        stride_o_h: tl.constexpr,

        HEAD_DIM: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        SPLIT_BLOCK: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)

        # Triton reductions are simplest over a power-of-two lane count.
        # SPLIT_BLOCK is next_power_of_2(NUM_SPLITS), supplied by the host.
        s = tl.arange(0, SPLIT_BLOCK)
        d = tl.arange(0, HEAD_DIM)
        split_mask = s < NUM_SPLITS

        pm = b * stride_pm_b + h * stride_pm_h + s * stride_pm_s
        m_s = tl.load(partial_m_ptr + pm, mask=split_mask, other=-float("inf")).to(tl.float32)
        l_s = tl.load(partial_l_ptr + pm, mask=split_mask, other=0.0).to(tl.float32)
        valid = split_mask & (l_s > 0.0)

        safe_m_s = tl.where(valid, m_s, -float("inf"))
        m = tl.max(safe_m_s, axis=0)
        m_for_exp = tl.where(valid, m_s, 0.0)
        # seq_lens are validated as positive by the host, so at least one
        # split is valid for every [request, head].
        alpha_raw = tl.exp(m_for_exp - m)
        alpha = tl.where(valid, alpha_raw, 0.0)
        denom = tl.sum(alpha * l_s, axis=0)

        po = (
            partial_o_ptr
            + b * stride_po_b
            + h * stride_po_h
            + s[:, None] * stride_po_s
            + d[None, :]
        )
        o_s = tl.load(po, mask=split_mask[:, None], other=0.0).to(tl.float32)
        numer = tl.sum(alpha[:, None] * o_s, axis=0)
        out = tl.where(denom > 0.0, numer / denom, 0.0)

        tl.store(out_ptr + b * stride_o_b + h * stride_o_h + d, out)

if triton is None:
    splitkv_int4_partials_kernel = None
    merge_splitkv_kernel = None
