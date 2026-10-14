"""Matched FP16 paged baseline; same online-softmax loop and merge as v2.

Signature retains unused scale/encoding arguments for an explicit structural
comparison. FP16 values are loaded directly; no unpack/dequant is performed.
"""
import triton
import triton.language as tl

if triton is not None:
    @triton.jit
    def splitkv_fp16_partials_kernel(
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
        """Matched FP16 Stage-1 split-KV decode kernel.

        Contract:
        q                  [B, Hq, D]
        FP16 K/V           [num_pages, Hkv, PAGE_SIZE, D]
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

            k_ptrs = (k_packed_ptr + phys_page[:, None] * stride_k_page
                      + kvh * stride_k_head + page_off[:, None] * stride_k_tok
                      + d[None, :])
            k = tl.load(k_ptrs, mask=tok_mask[:, None], other=0.0).to(tl.float32)

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

            v_ptrs = (v_packed_ptr + phys_page[:, None] * stride_v_page
                      + kvh * stride_v_head + page_off[:, None] * stride_v_tok
                      + d[None, :])
            v = tl.load(v_ptrs, mask=tok_mask[:, None], other=0.0).to(tl.float32)

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


