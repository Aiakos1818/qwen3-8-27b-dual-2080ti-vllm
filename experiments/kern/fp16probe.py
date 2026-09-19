import torch, re
from torch.utils.cpp_extension import load_inline
src = r"""
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#define D_HD 256
#define H_KV 2
#define NTHREADS 512

__device__ __forceinline__ __half e4m3_to_half(unsigned u) {
  const unsigned e = (u >> 3) & 0xFu;
  if (__builtin_expect(e == 0u, 0))
    return __float2half_rn((float)(u & 0x7u) * (1.0f / 512.0f));
  const unsigned bits = ((u & 0x80u) << 8) | ((e + 8u) << 10) | ((u & 0x7u) << 7);
  __half h; memcpy(&h, &bits, 2); return h;
}

// MODE 0: fp8 KV -> convert -> smem (TILE=32 rows)
// MODE 1: fp16 KV -> no conversion -> smem (TILE=16 rows, same elements)
template <int MODE>
__global__ void __launch_bounds__(NTHREADS, 1)
stage_kernel(const void* __restrict__ kv, const int* __restrict__ idx,
             int kv_len, int nchunks, int per, float* __restrict__ sink) {
  if (MODE == 0) {
    __shared__ __half sm[32][D_HD];
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    const int lo = chunk * per, hi = min(lo + per, kv_len);
    if (lo >= kv_len) return;
    const uint8_t* base = (const uint8_t*)kv;
    for (int t0 = lo; t0 < hi; t0 += 32) {
      // 8 fp8 per thread per chunk: 8B load, one 16B vector store (conflict free)
#pragma unroll
      for (int half_tile = 0; half_tile < 2; ++half_tile) {
        const int c = threadIdx.x + half_tile * NTHREADS;   // chunk id in tile
        const int row = c / 32, col8 = (c % 32) * 8;
        const int pid = idx[min(t0 / 16 + row / 16, 999999)];
        const uint8_t* p = base + ((long)pid * 2) * (16 * H_KV * D_HD) +
                           (long)(row % 16) * (H_KV * D_HD) + kvh * D_HD + col8;
        const uint2 a = *reinterpret_cast<const uint2*>(p);
        __half o[8];
#pragma unroll
        for (int j = 0; j < 8; ++j)
          o[j] = e4m3_to_half(j < 4 ? ((a.x >> (8 * j)) & 0xffu) : ((a.y >> (8 * (j - 4))) & 0xffu));
        *reinterpret_cast<uint4*>(&sm[row][col8]) = *reinterpret_cast<uint4*>(o);
      }
      __syncthreads();
    }
    if (__half2float(sm[threadIdx.x % 32][(threadIdx.x * 7) % D_HD]) == 12345.678f) sink[blockIdx.x] = 1.f;
  } else {
    __shared__ __half sm[16][D_HD];
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    const int lo = chunk * per, hi = min(lo + per, kv_len);
    if (lo >= kv_len) return;
    const uint16_t* base = (const uint16_t*)kv;
    for (int t0 = lo; t0 < hi; t0 += 16) {
      const int row = threadIdx.x / 32, c0 = (threadIdx.x % 32) * 8;
      const int pid = idx[min(t0 / 16 + row / 16, 999999)];
      const uint16_t* p = base + ((long)pid * 2 * 16) * (H_KV * D_HD) +
                          (long)(row % 16) * (H_KV * D_HD) + kvh * D_HD + c0;
      const uint4 a = *reinterpret_cast<const uint4*>(p);   // 8 fp16 = 16 B
      *reinterpret_cast<uint4*>(&sm[row][c0]) = a;          // vector store, no conversion
      __syncthreads();
    }
    if (__half2float(sm[threadIdx.x % 16][(threadIdx.x * 7) % D_HD]) == 12345.678f) sink[blockIdx.x] = 1.f;
  }
}

double run(torch::Tensor kv, torch::Tensor idx, int64_t kv_len, int64_t nchunks,
           int64_t per, int64_t mode) {
  auto sink = torch::empty({(long)nchunks, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(kv.device()));
  dim3 grid((unsigned)nchunks, H_KV);
  if (mode == 0) stage_kernel<0><<<grid, NTHREADS>>>(kv.data_ptr(), idx.data_ptr<int>(), kv_len, nchunks, per, sink.data_ptr<float>());
  else           stage_kernel<1><<<grid, NTHREADS>>>(kv.data_ptr(), idx.data_ptr<int>(), kv_len, nchunks, per, sink.data_ptr<float>());
  return 0.0;
}
"""
m = load_inline(name="fp16probe", cpp_sources="double run(torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t);",
                cuda_sources=src, functions=["run"],
                extra_cuda_cflags=["-O3","-arch=sm_75","--use_fast_math"], verbose=False)
dev="cuda:0"; torch.cuda.set_device(dev)
H_KV,D,PAGE,KV=2,256,16,250000
npages=(KV+PAGE-1)//PAGE+8
nchunks=68; per=64*((KV//64+67)//68)
idx=torch.arange(npages,dtype=torch.int32,device=dev)
kv8=torch.randint(0,255,(npages,2,PAGE,H_KV,D),dtype=torch.uint8,device=dev).view(torch.float8_e4m3fn)
kf16=torch.randint(0,32000,(npages,2,PAGE,H_KV,D),dtype=torch.int16,device=dev).view(torch.float16)
def t(fn,n=30):
    fn(); torch.cuda.synchronize(); a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record()
    for _ in range(n): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)/n
r8=t(lambda: m.run(kv8,idx,KV,nchunks,per,0))
r16=t(lambda: m.run(kf16,idx,KV,nchunks,per,1))
print(f"  fp8  KV: 读 128 MB + 转换 + 写 smem   {r8:6.3f} ms")
print(f"  fp16 KV: 读 256 MB + 无转换 + 写 smem {r16:6.3f} ms")
print(f"  → fp16 相对 fp8: {100*(r16/r8-1):+.1f}%")

# flashinfer 同 harness 对照（fp8 vs fp16 KV，250K，q_len=6）
import flashinfer
H_Q,QL=12,6
q=torch.randn(QL,H_Q,D,dtype=torch.float16,device=dev)*0.5
for fp8 in (True,False):
    kv=(kv8 if fp8 else kf16)
    qi=torch.tensor([0,QL],dtype=torch.int32,device=dev)
    pi=torch.tensor([0,npages],dtype=torch.int32,device=dev)
    pidx=torch.arange(npages,dtype=torch.int32,device=dev)
    last=torch.tensor([KV-(npages-1)*PAGE],dtype=torch.int32,device=dev)
    ws=torch.empty(512*1024*1024,dtype=torch.uint8,device=dev)
    w=flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws,kv_layout="NHD")
    dt=torch.float8_e4m3fn if fp8 else torch.float16
    w.plan(qi,pi,pidx,last,H_Q,H_KV,D,PAGE,head_dim_vo=D,causal=True,q_data_type=torch.float16,kv_data_type=dt)
    for _ in range(3): w.run(q,kv)
    torch.cuda.synchronize()
    st=torch.cuda.Event(True); en=torch.cuda.Event(True); st.record()
    for _ in range(20): w.run(q,kv)
    en.record(); torch.cuda.synchronize()
    print(f"  flashinfer {'fp8 ' if fp8 else 'fp16'} KV: {st.elapsed_time(en)/20:6.3f} ms")
