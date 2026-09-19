import torch
from torch.utils.cpp_extension import load_inline
src = r"""
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
using namespace nvcuda;
extern "C" __global__ void __launch_bounds__(512, 1) k1(const __half* in, float* out) {
  extern __shared__ char smem[];
  wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a;
  wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> b;
  wmma::fragment<wmma::accumulator,16,16,16,float> c;
  wmma::fill_fragment(c, 0.f);
  wmma::load_matrix_sync(a, in, 256);
  wmma::load_matrix_sync(b, in, 256);
  wmma::mma_sync(c, a, b, c);
  if (threadIdx.x == 0) out[blockIdx.x] = c.x[0] + smem[0];
}
int run(torch::Tensor in, torch::Tensor out, int64_t smem) {
  cudaError_t e1 = cudaFuncSetAttribute(k1, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  k1<<<2, 512, (int)smem>>>(  (const __half*)in.data_ptr(), (float*)out.data_ptr());
  cudaError_t e2 = cudaGetLastError();
  return (int)e1 * 100 + (int)e2;
}
"""
m = load_inline(name="mini", cpp_sources="int run(torch::Tensor, torch::Tensor, int64_t);",
                cuda_sources=src, functions=["run"],
                extra_cuda_cflags=["-O3","-arch=sm_75"], verbose=False)
inp = torch.randn(16, 256, dtype=torch.float16, device="cuda:0")
out = torch.zeros(2, device="cuda:0")
for smem in (0, 1024, 40960, 49152, 65536):
    r = m.run(inp, out, smem)
    print(f"  smem={smem:6d}  setattr*100+launch = {r}   (0 = OK)")
