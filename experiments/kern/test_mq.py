import time, sys, torch
import os
import triton
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mqattn import MQAttn

dev = "cuda:0"; torch.cuda.set_device(dev)
H_Q, H_KV, D, QL, PAGE = 12, 2, 256, 6, 16


def make_kv(kv_len, dtype, page=PAGE):
    npages = (kv_len + page - 1) // page
    kv = (torch.randn(npages, 2, page, H_KV, D, dtype=torch.float16, device=dev) * 0.1)
    if dtype == torch.float8_e4m3fn:
        kv = kv.to(dtype)
    return kv, npages


def ref(q, kv, kv_len, scale):
    k = kv[:, 0].reshape(-1, H_KV, D)[:kv_len].float()
    v = kv[:, 1].reshape(-1, H_KV, D)[:kv_len].float()
    out = torch.zeros(QL, H_Q, D, dtype=torch.float32, device=dev)
    for h in range(H_Q):
        kh = h // (H_Q // H_KV)
        s = (q[:, h].float() @ k[:, kh].T) * scale
        rows = torch.arange(QL, device=dev)[:, None]
        cols = torch.arange(kv_len, device=dev)[None, :]
        lim = kv_len - QL + rows + 1
        s = s.masked_fill(cols >= lim, float("-inf"))
        p = torch.softmax(s, dim=-1)
        out[:, h] = p @ v[:, kh]
    return out


def check(fp8=False, kv_len=4096):
    dtype = torch.float8_e4m3fn if fp8 else torch.float16
    q = (torch.randn(QL, H_Q, D, dtype=torch.float16, device=dev) * 0.5)
    kv, npages = make_kv(kv_len, dtype)
    idx = torch.arange(npages, dtype=torch.int32, device=dev)
    scale = 1.0 / (D ** 0.5)
    k = MQAttn(num_seg=8, block_n=64, num_warps=8, num_stages=1, dc=2)
    out = k.run(q, kv, idx, kv_len, scale).float()
    r = ref(q, kv, kv_len, scale)
    err = (out - r).abs().max().item()
    denom = r.abs().max().item()
    print(f"   fp8={fp8} kv_len={kv_len}: 最大绝对误差 {err:.5f} (参考幅值 {denom:.3f})  相对 {err/denom:.2e}")
    return err / denom


def bench(kv_len=250000, fp8=True, label=""):
    dtype = torch.float8_e4m3fn if fp8 else torch.float16
    q = (torch.randn(QL, H_Q, D, dtype=torch.float16, device=dev) * 0.5)
    kv, npages = make_kv(kv_len, dtype)
    idx = torch.arange(npages, dtype=torch.int32, device=dev)
    scale = 1.0 / (D ** 0.5)
    nbytes = npages * 2 * PAGE * H_KV * D * (1 if fp8 else 2)
    print(f"   --- {label} kv_len={kv_len} fp8={fp8} KV 读量 {nbytes/1e6:.0f} MB ---")
    for num_seg, bn, nw, ns, dc in CFGS:
        k = MQAttn(num_seg=num_seg, block_n=bn, num_warps=nw, num_stages=ns, dc=dc)
        try:
            o = k.run(q, kv, idx, kv_len, scale)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"     seg={num_seg:3d} bn={bn:3d} w={nw:2d} dc={dc} 失败: {type(e).__name__} {str(e)[:70]}")
            continue
        for _ in range(3):
            k.run(q, kv, idx, kv_len, scale)
        torch.cuda.synchronize()
        n = 20
        st = torch.cuda.Event(True); en = torch.cuda.Event(True)
        st.record()
        for _ in range(n):
            k.run(q, kv, idx, kv_len, scale)
        en.record(); torch.cuda.synchronize()
        ms = st.elapsed_time(en) / n
        print(f"     seg={num_seg:3d} bn={bn:3d} w={nw:2d} dc={dc}  {ms:.3f} ms  {nbytes/ms/1e6:.0f} GB/s")
    return q, kv, idx, scale


CFGS = [(8, 64, 8, 1, 2), (16, 64, 8, 1, 2), (32, 64, 8, 1, 2), (64, 64, 8, 1, 2),
        (32, 32, 8, 1, 2), (64, 32, 8, 1, 2), (32, 64, 4, 1, 2), (64, 64, 16, 1, 2),
        (32, 64, 8, 1, 4), (64, 64, 8, 1, 4), (32, 128, 8, 1, 2)]

if __name__ == "__main__":
    print("  == 正确性（fp16 / fp8）==")
    check(False, 4096)
    check(True, 4096)
    print("  == 速度：我的 kernel ===")
    bench(250000, fp8=True, label="Triton mq")
    bench(250000, fp8=False, label="Triton mq")
    print("  == 速度：flashinfer 参照 ==")
    import flashinfer
    for fp8 in (True, False):
        dtype = torch.float8_e4m3fn if fp8 else torch.float16
        q = (torch.randn(QL, H_Q, D, dtype=torch.float16, device=dev) * 0.5)
        kv, npages = make_kv(250000, dtype)
        qi = torch.tensor([0, QL], dtype=torch.int32, device=dev)
        pi = torch.tensor([0, npages], dtype=torch.int32, device=dev)
        pidx = torch.arange(npages, dtype=torch.int32, device=dev)
        last = torch.tensor([250000 - (npages - 1) * PAGE], dtype=torch.int32, device=dev)
        ws = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=dev)
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, kv_layout="NHD")
        w.plan(qi, pi, pidx, last, H_Q, H_KV, D, PAGE, head_dim_vo=D, causal=True,
               q_data_type=torch.float16, kv_data_type=dtype)
        for _ in range(3):
            w.run(q, kv)
        torch.cuda.synchronize()
        st = torch.cuda.Event(True); en = torch.cuda.Event(True)
        st.record()
        for _ in range(20):
            w.run(q, kv)
        en.record(); torch.cuda.synchronize()
        ms = st.elapsed_time(en) / 20
        nb = npages * 2 * PAGE * H_KV * D * (1 if fp8 else 2)
        print(f"     flashinfer fp8={fp8}  {ms:.3f} ms  {nb/ms/1e6:.0f} GB/s")
