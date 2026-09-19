import sys, torch
import os
from torch.utils.cpp_extension import load_inline

cpp = """
#include <c10/cuda/CUDAStream.h>
void mq_attn_launch_raw(const void*, const void*, const void*, void*, void*, int, int, int, int, int, void*);
void mq_merge_launch_raw(const void*, const void*, void*, int, void*);
void mq_readonly_launch(const void*, const void*, int, int, int, void*, void*);
int mq_smem_bytes();
const char* mq_last_error();
int mq_dbg(int);
int mq_try_setattr(int);
int mq_try_launch(int);
int mq_merge_err();

void mq_attn_run(torch::Tensor q, torch::Tensor kv, torch::Tensor idx,
                 torch::Tensor part_o, torch::Tensor part_lse,
                 int64_t kv_len, int64_t nchunks, int64_t per, int64_t mode, int64_t smem_ovr) {
  mq_attn_launch_raw(q.data_ptr(), kv.data_ptr(), idx.data_ptr(),
                     part_o.data_ptr(), part_lse.data_ptr(),
                     (int)kv_len, (int)nchunks, (int)per, (int)mode, (int)smem_ovr,
                     (void*)c10::cuda::getCurrentCUDAStream());
}
void mq_readonly_run(torch::Tensor kv, torch::Tensor idx, int64_t kv_len, int64_t nchunks,
                     int64_t per, torch::Tensor sink) {
  mq_readonly_launch(kv.data_ptr(), idx.data_ptr(), (int)kv_len, (int)nchunks, (int)per,
                     sink.data_ptr(), (void*)c10::cuda::getCurrentCUDAStream());
}
void mq_merge_run(torch::Tensor part_o, torch::Tensor part_lse, torch::Tensor out,
                  int64_t nchunks) {
  mq_merge_launch_raw(part_o.data_ptr(), part_lse.data_ptr(), out.data_ptr(),
                      (int)nchunks, (void*)c10::cuda::getCurrentCUDAStream());
}
"""
cuda = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mq_attn.cu")).read()
mod = load_inline(name="mqcuda", cpp_sources=cpp, cuda_sources=cuda,
                  functions=["mq_attn_run", "mq_merge_run", "mq_readonly_run", "mq_smem_bytes", "mq_last_error", "mq_dbg", "mq_try_setattr", "mq_try_launch", "mq_merge_err"],
                  extra_cuda_cflags=["-O3", "-arch=sm_75", "--use_fast_math"],
                  verbose=False)

dev = "cuda:0"; torch.cuda.set_device(dev)
H_Q, H_KV, D, QL, PAGE = 12, 2, 256, 6, 16
print(f"  smem = {mod.mq_smem_bytes()/1024:.1f} KB")
print(f"   setattr({mod.mq_smem_bytes()}) -> {mod.mq_try_setattr(mod.mq_smem_bytes())}")


def run(q, kv, idx, kv_len, nchunks, per, mode=0, smem_ovr=0):
    part_o = torch.zeros(nchunks * H_KV * 48 * D, dtype=torch.float32, device=dev)
    part_lse = torch.zeros(nchunks * H_KV * 48 * 2, dtype=torch.float32, device=dev)
    out = torch.zeros(QL, H_Q, D, dtype=torch.float16, device=dev)
    mod.mq_attn_run(q, kv, idx, part_o, part_lse, kv_len, nchunks, per, mode, smem_ovr)
    print(f"   [dbg] setattr={mod.mq_dbg(0)} max_smem_optin={mod.mq_dbg(1)} launch_err={mod.mq_dbg(2)} smem={mod.mq_dbg(3)}")
    mod.mq_merge_run(part_o, part_lse, out, nchunks)
    torch.cuda.synchronize()
    err = mod.mq_last_error()
    if err != "no error":
        print(f"   !! CUDA 错误: {err}")
    return out


def ref(q, kv, kv_len):
    k = kv[:, 0].reshape(-1, H_KV, D)[:kv_len].float()
    v = kv[:, 1].reshape(-1, H_KV, D)[:kv_len].float()
    out = torch.zeros(QL, H_Q, D, dtype=torch.float32, device=dev)
    rows = torch.arange(QL, device=dev)[:, None]
    cols = torch.arange(kv_len, device=dev)[None, :]
    lim = kv_len - QL + rows + 1
    for h in range(H_Q):
        kh = h // (H_Q // H_KV)
        s = (q[:, h].float() @ k[:, kh].T) * (1.0 / 16.0)
        p = torch.softmax(s.masked_fill(cols >= lim, float("-inf")), dim=-1)
        out[:, h] = p @ v[:, kh]
    return out


torch.manual_seed(0)
print("  == 正确性（kv_len 512, 2 chunks）==")
KV = 512
kv = torch.randn(KV // PAGE + 1, 2, PAGE, H_KV, D, dtype=torch.float16, device=dev) * 0.1
kv = kv.to(torch.float8_e4m3fn)
idx = torch.arange(kv.shape[0], dtype=torch.int32, device=dev)
q = torch.randn(QL, H_Q, D, dtype=torch.float16, device=dev) * 0.5
got = run(q, kv, idx, KV, 2, 256).float()
r = ref(q, kv, KV)
rel = (got - r).abs().max().item() / r.abs().max().item()
print(f"   max|err| {((got-r).abs().max().item()):.4f}  rel {rel:.2e}  {'OK' if rel < 3e-2 else 'FAIL'}")

print("  == 速度（kv_len 250000, 68 chunks）==")
KV = 250000
npages = (KV + PAGE - 1) // PAGE
per = ((npages + 68 - 1) // 68) * PAGE
nchunks = (npages * PAGE + per - 1) // per
kv = torch.randint(0, 200, (npages, 2, PAGE, H_KV, D), dtype=torch.uint8, device=dev)
idx = torch.arange(npages, dtype=torch.int32, device=dev)
q = torch.randn(QL, H_Q, D, dtype=torch.float16, device=dev) * 0.5
part_o = torch.zeros(nchunks * H_KV * 48 * D, dtype=torch.float32, device=dev)
part_lse = torch.zeros(nchunks * H_KV * 48 * 2, dtype=torch.float32, device=dev)
out = torch.zeros(QL, H_Q, D, dtype=torch.float16, device=dev)
def timeit(fn, n=20):
    fn(); torch.cuda.synchronize()
    st = torch.cuda.Event(True); en = torch.cuda.Event(True); st.record()
    for _ in range(n): fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / n
t_attn = timeit(lambda: mod.mq_attn_run(q, kv, idx, part_o, part_lse, KV, nchunks, per, 0, 0))
for md, sm, nm in ((6,0,"probe pat, smem=0"), (6,1024,"probe pat, smem=1K"),
                   (6,45952,"probe pat, smem=45K"), (4,0,"loads only, smem=0"),
                   (0,45952,"full, smem=45K")):
    t = timeit(lambda: mod.mq_attn_run(q, kv, idx, part_o, part_lse, KV, nchunks, per, md, sm))
    print(f"   mode={md} smem={sm:6d} {nm:20s} {t:7.3f} ms")
t_merge = timeit(lambda: mod.mq_merge_run(part_o, part_lse, out, nchunks))
nb = npages * 2 * PAGE * H_KV * D
print(f"   per={per} nchunks={nchunks}")
print(f"   attn  {t_attn:7.3f} ms  {nb*1.5/t_attn/1e6:6.0f} GB/s (读 {nb*1.5/1e6:.0f} MB)")
print(f"   merge {t_merge:7.3f} ms")
print(f"   total {t_attn+t_merge:7.3f} ms")
