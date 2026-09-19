import sys, torch
import os
import triton
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mqattn import MQAttn

dev = "cuda:0"; torch.cuda.set_device(dev)
H_Q, H_KV, D, QL, PAGE = 12, 2, 256, 6, 16


def make_kv(kv_len, dtype, page=PAGE):
    npages = (kv_len + page - 1) // page
    kv = torch.randn(npages, 2, page, H_KV, D, dtype=torch.float16, device=dev) * 0.1
    return kv.to(dtype), npages


def ref(q, kv, kv_len, scale):
    k = kv[:, 0].reshape(-1, H_KV, D)[:kv_len].float()
    v = kv[:, 1].reshape(-1, H_KV, D)[:kv_len].float()
    out = torch.zeros(QL, H_Q, D, dtype=torch.float32, device=dev)
    rows = torch.arange(QL, device=dev)[:, None]
    cols = torch.arange(kv_len, device=dev)[None, :]
    lim = kv_len - QL + rows + 1
    for h in range(H_Q):
        kh = h // (H_Q // H_KV)
        s = (q[:, h].float() @ k[:, kh].T) * scale
        p = torch.softmax(s.masked_fill(cols >= lim, float("-inf")), dim=-1)
        out[:, h] = p @ v[:, kh]
    return out


scale = 1.0 / (D ** 0.5)

# (num_seg, block_n, num_warps, num_stages, dc)
CFGS = [
    (16, 16, 8, 1, 1), (32, 16, 8, 1, 1), (16, 32, 8, 1, 1), (32, 32, 8, 1, 1),
    (16, 16, 16, 1, 1), (32, 32, 16, 1, 1), (16, 16, 8, 1, 2), (32, 16, 8, 1, 2),
    (16, 32, 8, 1, 2), (32, 32, 8, 1, 2), (32, 16, 16, 1, 2), (64, 16, 8, 1, 2),
]

print(f"  {'seg':>4} {'bn':>3} {'w':>3} {'dc':>3} | {'小 KV 正确性':>22} | {'250K: ms':>9} {'GB/s':>6}")
for num_seg, bn, nw, ns, dc in CFGS:
    tag = f"{num_seg:>4} {bn:>3} {nw:>3} {dc:>3}"

    # --- correctness on a short KV (fp16) ---
    qs = torch.randn(QL, H_Q, D, dtype=torch.float16, device=dev) * 0.5
    kvs, np_ = make_kv(4096, torch.float16)
    idxs = torch.arange(np_, dtype=torch.int32, device=dev)
    try:
        k = MQAttn(num_seg=num_seg, block_n=bn, num_warps=nw, num_stages=ns, dc=dc)
        got = k.run(qs, kvs, idxs, 4096, scale).float()
        r = ref(qs, kvs, 4096, scale)
        rel = (got - r).abs().max().item() / max(r.abs().max().item(), 1e-6)
        cstr = f"rel_err {rel:.2e} {'OK' if rel < 2e-2 else 'FAIL'}"
    except Exception as e:
        print(f"  {tag} | {'编译失败: ' + type(e).__name__ + ' ' + str(e)[:40]:>22} |")
        continue

    # --- speed on 250K, fp8 KV ---
    q = torch.randn(QL, H_Q, D, dtype=torch.float16, device=dev) * 0.5
    kv, npages = make_kv(250000, torch.float8_e4m3fn)
    idx = torch.arange(npages, dtype=torch.int32, device=dev)
    try:
        for _ in range(2):
            k.run(q, kv, idx, 250000, scale)
        torch.cuda.synchronize()
        st = torch.cuda.Event(True); en = torch.cuda.Event(True)
        st.record()
        for _ in range(20):
            k.run(q, kv, idx, 250000, scale)
        en.record(); torch.cuda.synchronize()
        ms = st.elapsed_time(en) / 20
        nb = npages * 2 * PAGE * H_KV * D
        print(f"  {tag} | {cstr:>22} | {ms:>9.3f} {nb/ms/1e6:>6.0f}")
    except Exception as e:
        print(f"  {tag} | {cstr:>22} | 运行失败: {type(e).__name__} {str(e)[:40]}")
