"""Multi-query (spec-verify shaped) segmented decode attention for SM75.

Why: a verify batch (q_len = 1+n) is treated as prefill by both the flashinfer
and the Triton backend, and flashinfer's prefill kernel then runs with one 64 KB
smem CTA per SM and only 4 warps.  Turing cannot hide DRAM latency with 4 warps,
so it reaches ~165 GB/s instead of ~500.

The traffic is fixed (each KV chunk must be read once per kv head, and the GQA
group shares it), so the only lever is occupancy: keep the tile small enough
that several CTAs fit per SM, and use enough warps.  Here the M dimension packs
all GROUP query heads x QL query rows (36 -> 64), the KV is split into NUM_SEG
segments, and head_dim is processed as DC chunks of DB so the Q tile stays small.
Partial outputs are merged by mq_merge_kernel.

flashinfer "NHD" paged KV layout:
  q:   [QL, H_Q, D] fp16
  kv:  [num_pages, 2, PAGE, H_KV, D]  fp16 or fp8_e4m3
  idx: [num_pages] int32
"""

import torch
import triton
import triton.language as tl


@triton.jit
def mq_seg_kernel(
    q_ptr, kv_ptr, idx_ptr,
    pacc_ptr, plse_ptr,
    kv_len,
    H_Q: tl.constexpr, H_KV: tl.constexpr, GROUP: tl.constexpr,
    D: tl.constexpr, DB: tl.constexpr, DC: tl.constexpr, QL: tl.constexpr,
    PAGE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    NUM_SEG: tl.constexpr, SM_SCALE: tl.constexpr,
):
    seg = tl.program_id(0)
    kvh = tl.program_id(1)

    chunk = (kv_len + NUM_SEG - 1) // NUM_SEG
    lo = seg * chunk
    hi = tl.minimum(lo + chunk, kv_len)

    offs_m = tl.arange(0, BLOCK_M)
    m_ok = offs_m < GROUP * QL
    q_row = offs_m % QL
    head_in_grp = offs_m // QL
    qo_head = kvh * GROUP + head_in_grp
    lim = kv_len - QL + q_row + 1

    offs_d = tl.arange(0, DB)
    qb = (q_row[:, None] * H_Q + qo_head[:, None]) * D
    q0 = tl.load(q_ptr + qb + offs_d[None, :], mask=m_ok[:, None], other=0.0)
    if DC == 2:
        q1 = tl.load(q_ptr + qb + DB + offs_d[None, :], mask=m_ok[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc0 = tl.zeros([BLOCK_M, DB], tl.float32)
    if DC == 2:
        acc1 = tl.zeros([BLOCK_M, DB], tl.float32)

    kv_plane = PAGE * H_KV * D
    for n0 in range(lo, hi, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_ok = offs_n < hi
        page = offs_n // PAGE
        slot = offs_n % PAGE
        pidx = tl.load(idx_ptr + tl.where(n_ok, page, 0), mask=n_ok, other=0)
        base = (pidx * 2 * PAGE + slot) * (H_KV * D) + kvh * D

        k0 = tl.load(kv_ptr + base[:, None] + offs_d[None, :], mask=n_ok[:, None], other=0.0)
        s = tl.dot(q0, tl.trans(k0.to(tl.float16))) * SM_SCALE
        if DC == 2:
            k1 = tl.load(kv_ptr + base[:, None] + DB + offs_d[None, :], mask=n_ok[:, None], other=0.0)
            s = s + tl.dot(q1, tl.trans(k1.to(tl.float16))) * SM_SCALE

        keep = n_ok[None, :] & (offs_n[None, :] < lim[:, None])
        s = tl.where(keep, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        p16 = p.to(tl.float16)

        v0 = tl.load(kv_ptr + base[:, None] + offs_d[None, :] + kv_plane,
                     mask=n_ok[:, None], other=0.0)
        acc0 = acc0 * alpha[:, None] + tl.dot(p16, v0.to(tl.float16))
        if DC == 2:
            v1 = tl.load(kv_ptr + base[:, None] + DB + offs_d[None, :] + kv_plane,
                         mask=n_ok[:, None], other=0.0)
            acc1 = acc1 * alpha[:, None] + tl.dot(p16, v1.to(tl.float16))
        m_i = m_new

    o_off = (seg * H_KV + kvh) * BLOCK_M * D
    rows = offs_m[:, None] * D + offs_d[None, :]
    tl.store(pacc_ptr + o_off + rows, acc0)
    if DC == 2:
        tl.store(pacc_ptr + o_off + rows + DB, acc1)
    ls_off = (seg * H_KV + kvh) * BLOCK_M
    lse_base = NUM_SEG * H_KV * BLOCK_M
    tl.store(plse_ptr + ls_off + offs_m, m_i)
    tl.store(plse_ptr + lse_base + ls_off + offs_m, l_i)


@triton.jit
def mq_merge_kernel(
    pacc_ptr, plse_ptr, out_ptr,
    H_Q: tl.constexpr, H_KV: tl.constexpr, GROUP: tl.constexpr,
    D: tl.constexpr, QL: tl.constexpr, BLOCK_M: tl.constexpr, NUM_SEG: tl.constexpr,
):
    kvh = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    lse_base = NUM_SEG * H_KV * BLOCK_M

    m_g = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_g = tl.zeros([BLOCK_M], tl.float32)
    for s in range(NUM_SEG):
        ls_off = (s * H_KV + kvh) * BLOCK_M
        m_s = tl.load(plse_ptr + ls_off + offs_m)
        l_s = tl.load(plse_ptr + lse_base + ls_off + offs_m)
        m_g = tl.maximum(m_g, m_s)
        l_g = l_g + l_s * tl.exp(m_s - m_g)

    acc = tl.zeros([BLOCK_M, D], tl.float32)
    for s in range(NUM_SEG):
        ls_off = (s * H_KV + kvh) * BLOCK_M
        m_s = tl.load(plse_ptr + ls_off + offs_m)
        w = tl.exp(m_s - m_g)
        o_off = (s * H_KV + kvh) * BLOCK_M * D
        acc += tl.load(pacc_ptr + o_off + offs_m[:, None] * D + offs_d[None, :]) * w[:, None]

    q_row = offs_m % QL
    qo_head = kvh * GROUP + offs_m // QL
    dst = (q_row[:, None] * H_Q + qo_head[:, None]) * D + offs_d[None, :]
    tl.store(out_ptr + dst, acc / l_g[:, None], mask=(offs_m < GROUP * QL)[:, None])


class MQAttn:
    def __init__(self, num_seg=16, block_n=64, num_warps=8, num_stages=1, dc=2):
        self.num_seg = num_seg
        self.block_n = block_n
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.dc = dc
        self._key = None

    def _ensure(self, ql, hq, h_kv, d, dev):
        key = (ql, hq, h_kv, d, str(dev))
        if self._key == key:
            return self._bm, self._pacc, self._plse
        self._bm = triton.next_power_of_2((hq // h_kv) * ql)
        self._pacc = torch.empty(self.num_seg * h_kv * self._bm * d, dtype=torch.float32, device=dev)
        self._plse = torch.empty(2 * self.num_seg * h_kv * self._bm, dtype=torch.float32, device=dev)
        self._key = key
        return self._bm, self._pacc, self._plse

    def run(self, q, kv, idx, kv_len, scale=None):
        ql, hq, d = q.shape
        h_kv = kv.shape[3]
        db = d // self.dc
        bm, pacc, plse = self._ensure(ql, hq, h_kv, d, q.device)
        out = torch.empty_like(q)
        sm = 1.0 / (d ** 0.5) if scale is None else scale
        mq_seg_kernel[(self.num_seg, h_kv)](
            q, kv, idx, pacc, plse, kv_len,
            H_Q=hq, H_KV=h_kv, GROUP=hq // h_kv, D=d, DB=db, DC=self.dc, QL=ql,
            PAGE=kv.shape[2], BLOCK_M=bm, BLOCK_N=self.block_n, NUM_SEG=self.num_seg,
            SM_SCALE=sm, num_warps=self.num_warps, num_stages=self.num_stages,
        )
        mq_merge_kernel[(h_kv,)](
            pacc, plse, out,
            H_Q=hq, H_KV=h_kv, GROUP=hq // h_kv, D=d, QL=ql,
            BLOCK_M=bm, NUM_SEG=self.num_seg, num_warps=4,
        )
        return out
