"""Bandwidth probe for the exact paged-fp8-KV access pattern the long-context
decode needs, written as a raw CUDA kernel so it is not bound by flashinfer's
4-warp-per-CTA layout.

Reads K and V of the whole 250K-token cache (256 MB) with uint4 (16 x fp8)
vector loads, converts e4m3 -> fp16 in software (Turing has no fp8 convert),
and folds the result into a register accumulator so nothing is dead-code
eliminated.  Reports effective GB/s for a range of occupancies/chunk counts.
"""

import os, time
import torch
from torch.utils.cpp_extension import load_inline

src = r"""
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// e4m3 -> fp16 without hardware support (Turing).  Normal path is pure bit
// manipulation; subnormals (exp == 0) are only 7 values, handled in fp32.
__device__ __forceinline__ __half e4m3_to_half(unsigned u) {
  unsigned s = (u >> 7) & 1u;
  unsigned e = (u >> 3) & 0xFu;
  unsigned m = u & 0x7u;
  unsigned bits = (s << 15) | ((e + 8u) << 10) | (m << 7);
  __half hn;
  memcpy(&hn, &bits, 2);
  __half hs = __float2half_rn((float)m * (1.0f / 512.0f));
  return e == 0u ? hs : hn;
}

template <int THREADS>
__global__ void probe_kernel(const uint8_t* __restrict__ kv,
                             const int* __restrict__ idx,
                             int kv_len, int H_KV, int D, int PAGE,
                             int nchunks, float* __restrict__ sink) {
  const int chunk = blockIdx.x;
  const int kvh = blockIdx.y;
  const int per = (kv_len + nchunks - 1) / nchunks;
  const int lo = chunk * per;
  const int hi = min(lo + per, kv_len);
  if (lo >= kv_len) return;
  const int vecs_per_row = D / 16;               // uint4 = 16 fp8
  const long plane = (long)PAGE * H_KV * D;

  float acc = 0.f;
  for (int row = lo + (threadIdx.x / vecs_per_row); row < hi;
       row += THREADS / vecs_per_row) {
    const int col = (threadIdx.x % vecs_per_row) * 16;
    const int page = row / PAGE;
    const int slot = row % PAGE;
    const int pid = idx[page];
    const long base = ((long)pid * 2 * PAGE + slot) * (H_KV * (long)D) + kvh * D + col;
    const uint4* kp = reinterpret_cast<const uint4*>(kv + base);
    const uint4* vp = reinterpret_cast<const uint4*>(kv + base + plane);
    uint4 k = *kp, v = *vp;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      unsigned ku = reinterpret_cast<unsigned*>(&k)[i];
      unsigned vu = reinterpret_cast<unsigned*>(&v)[i];
      acc += __half2float(e4m3_to_half(ku & 0xffu)) +
             __half2float(e4m3_to_half((ku >> 8) & 0xffu)) +
             __half2float(e4m3_to_half((ku >> 16) & 0xffu)) +
             __half2float(e4m3_to_half((ku >> 24) & 0xffu));
      acc += __half2float(e4m3_to_half(vu & 0xffu)) +
             __half2float(e4m3_to_half((vu >> 8) & 0xffu)) +
             __half2float(e4m3_to_half((vu >> 16) & 0xffu)) +
             __half2float(e4m3_to_half((vu >> 24) & 0xffu));
    }
  }
  if (acc == 12345.678f) sink[blockIdx.x] = acc;   // never true; keeps acc live
}

double probe(torch::Tensor kv, torch::Tensor idx, int64_t kv_len, int64_t h_kv,
             int64_t d, int64_t page, int64_t nchunks, int64_t threads) {
  auto sink = torch::empty({(long)nchunks, 1},
                           torch::TensorOptions().dtype(torch::kFloat32).device(kv.device()));
  dim3 grid((unsigned)nchunks, (unsigned)h_kv);
  const uint8_t* kp = kv.data_ptr<uint8_t>();
  const int* ip = idx.data_ptr<int>();
  if (threads == 128)      probe_kernel<128><<<grid, 128>>>(kp, ip, kv_len, h_kv, d, page, nchunks, sink.data_ptr<float>());
  else if (threads == 256) probe_kernel<256><<<grid, 256>>>(kp, ip, kv_len, h_kv, d, page, nchunks, sink.data_ptr<float>());
  else                     probe_kernel<512><<<grid, 512>>>(kp, ip, kv_len, h_kv, d, page, nchunks, sink.data_ptr<float>());
  return 0.0;
}
"""

PROBE_SRC = None
mod = load_inline(
    name="kv_probe",
    cpp_sources="double probe(torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);",
    cuda_sources=src,
    functions=["probe"],
    extra_cuda_cflags=["-O3", "-arch=sm_75", "--use_fast_math"],
    verbose=False,
)

