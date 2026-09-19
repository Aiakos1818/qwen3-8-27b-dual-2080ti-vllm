import torch
from torch.utils.cpp_extension import load_inline
src = r"""
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#define D_HD 256
#define H_KV 2
#define TILE 32
#define NTHREADS 512

__device__ __forceinline__ __half e4m3_to_half(unsigned u) {
  const unsigned e = (u >> 3) & 0xFu;
  if (__builtin_expect(e == 0u, 0))
    return __float2half_rn((float)(u & 0x7u) * (1.0f / 512.0f));
  const unsigned bits = ((u & 0x80u) << 8) | ((e + 8u) << 10) | ((u & 0x7u) << 7);
  __half h; memcpy(&h, &bits, 2); return h;
}

// A: read + convert + accumulate in registers (no smem)   [like the fast probe]
// B: read + convert + write to smem (staging)             [like the real kernel]
template <int MODE>
__global__ void __launch_bounds__(NTHREADS, 1)
stage_kernel(const uint8_t* __restrict__ kv, const int* __restrict__ idx,
             int kv_len, int nchunks, int per, float* __restrict__ sink) {
  __shared__ __half sm[TILE][D_HD];
  const int chunk = blockIdx.x, kvh = blockIdx.y;
  const int lo = chunk * per, hi = min(lo + per, kv_len);
  if (lo >= kv_len) return;
  float acc = 0.f;
  for (int t0 = lo; t0 < hi; t0 += TILE) {
    const int row = threadIdx.x / 16, c0 = (threadIdx.x % 16) * 16;
    const int page = min(t0 / 16 + row / 16, 15 + nchunks * 0);
    const int pid = idx[t0 / 16 + row / 16];
    const uint8_t* p = kv + ((long)pid * 2) * (16 * H_KV * D_HD) +
                       (long)(row % 16) * (H_KV * D_HD) + kvh * D_HD + c0;
    const uint4 a = *reinterpret_cast<const uint4*>(p);
    const unsigned uu[4] = {a.x, a.y, a.z, a.w};
    if (MODE == 0) {
#pragma unroll
      for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int j = 0; j < 4; ++j)
          acc += __half2float(e4m3_to_half((uu[q] >> (8 * j)) & 0xffu));
    } else {
#pragma unroll
      for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int j = 0; j < 4; ++j)
          sm[row][c0 + q * 4 + j] = e4m3_to_half((uu[q] >> (8 * j)) & 0xffu);
    }
    __syncthreads();
  }
  if (MODE == 1) acc = __half2float(sm[threadIdx.x % TILE][(threadIdx.x * 7) % D_HD]);
  if (acc == 12345.678f) sink[blockIdx.x] = acc;
}

double run(torch::Tensor kv, torch::Tensor idx, int64_t kv_len, int64_t nchunks, int64_t per,
           int64_t mode) {
  auto sink = torch::empty({(long)nchunks, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(kv.device()));
  dim3 grid((unsigned)nchunks, H_KV);
  if (mode == 0) stage_kernel<0><<<grid, NTHREADS>>>(kv.data_ptr<uint8_t>(), idx.data_ptr<int>(), kv_len, nchunks, per, sink.data_ptr<float>());
  else           stage_kernel<1><<<grid, NTHREADS>>>(kv.data_ptr<uint8_t>(), idx.data_ptr<int>(), kv_len, nchunks, per, sink.data_ptr<float>());
  return 0.0;
}
"""
m = load_inline(name="stageprobe", cpp_sources="double run(torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t);",
                cuda_sources=src, functions=["run"],
                extra_cuda_cflags=["-O3","-arch=sm_75","--use_fast_math"], verbose=False)
dev="cuda:0"; torch.cuda.set_device(dev)
H_KV,D,PAGE,KV,TILE=2,256,16,250000,32
npages=(KV+PAGE-1)//PAGE+8
kv=torch.randint(0,200,(npages,2,PAGE,H_KV,D),dtype=torch.uint8,device=dev)
idx=torch.arange(npages,dtype=torch.int32,device=dev)
nchunks=68; per=64*((KV//64+67)//68)
def t(fn,n=30):
    fn(); torch.cuda.synchronize(); a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record()
    for _ in range(n): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)/n
r0=t(lambda: m.run(kv,idx,KV,nchunks,per,0))
r1=t(lambda: m.run(kv,idx,KV,nchunks,per,1))
kb=KV*H_KV*D   # K 只读 128 MB
print(f"  只读K+转换(寄存器)   {r0:6.3f} ms  {kb/r0/1e6:5.0f} GB/s")
print(f"  只读K+转换+写smem    {r1:6.3f} ms  {kb/r1/1e6:5.0f} GB/s")
