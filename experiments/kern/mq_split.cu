// Split multi-query paged attention for SM75 with fp8_e4m3 KV.
//
// Two small kernels instead of one big one, because ptxas generates badly
// scheduled memory code for the read loop when it lives inside the combined
// kernel (measured 0.43 ms standalone vs 3.36 ms inside the big kernel for the
// identical loop).  Both kernels here stay simple enough to schedule well.
//
//   A) mq_scores_kernel  : K only -> scaled+masked scores S[chunk][kvh][rows][keys]
//                          and the per-chunk row max
//   B) mq_pv_kernel      : S + V -> P = exp(S - m), O += P@V (wmma), l = sum P
//   merge (unchanged)    : combine chunks by log-sum-exp
//
// Traffic: read K (128 MB) + write S (36 MB) + read S (36 MB) + read V (128 MB)
// = 328 MB vs the ideal 256 MB, i.e. 1.28x, and both loops are simple.

#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>

using namespace nvcuda;

#define D_HD 256
#define Q_LEN 6
#define GROUP 6
#define H_KV 2
#define H_Q 12
#define M_REAL (Q_LEN * GROUP)   // 36
#define M_ROWS 48                // 36 padded to 3 x 16
#define NMT 3
#define NTHREADS 512
#define BLOCK_A 32               // keys per tile in kernel A
#define BLOCK_B 32               // keys per tile in kernel B

__device__ __forceinline__ __half e4m3_to_half(unsigned u) {
  const unsigned e = (u >> 3) & 0xFu;
  if (__builtin_expect(e == 0u, 0))   // subnormal / zero: rare, slow path only then
    return __float2half_rn((float)(u & 0x7u) * (1.0f / 512.0f));
  const unsigned bits = ((u & 0x80u) << 8) | ((e + 8u) << 10) | ((u & 0x7u) << 7);
  __half h;
  memcpy(&h, &bits, 2);
  return h;
}

__device__ __forceinline__ void unpack32(const uint8_t* __restrict__ p, __half* dst) {
  const uint4 a = *reinterpret_cast<const uint4*>(p);
  const uint4 b = *reinterpret_cast<const uint4*>(p + 16);
  unsigned u[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
  for (int q = 0; q < 8; ++q)
#pragma unroll
    for (int j = 0; j < 4; ++j)
      dst[q * 4 + j] = e4m3_to_half((u[q] >> (8 * j)) & 0xffu);
}

// ---------------------------------------------------------------------------
// kernel A: scores + row max
// ---------------------------------------------------------------------------
struct SmemA {
  __half q[M_ROWS][D_HD];      // 24 KB
  __half k[BLOCK_A][D_HD];     // 32 KB
  float s[NMT][16][BLOCK_A];   // 6 KB (wmma store wants fp32)
  float mrun[M_ROWS];
};

extern "C" __global__ void __launch_bounds__(NTHREADS, 1)
mq_scores_kernel(const __half* __restrict__ q, const uint8_t* __restrict__ kv,
                 const int* __restrict__ idx, __half* __restrict__ sbuf,
                 float* __restrict__ mchunk, int kv_len, int nchunks, int per, int npages) {
  extern __shared__ char raw[];
  SmemA& s = *reinterpret_cast<SmemA*>(raw);
  const int chunk = blockIdx.x, kvh = blockIdx.y;
  const int warp = threadIdx.x >> 5;

  for (int i = threadIdx.x; i < M_ROWS * D_HD; i += NTHREADS) {
    const int r = i / D_HD, d = i % D_HD;
    __half v = __float2half_rn(0.f);
    if (r < M_REAL) {
      const int qr = r / GROUP, hig = r % GROUP;
      v = q[(qr * H_Q + kvh * GROUP + hig) * D_HD + d];
    }
    s.q[r][d] = v;
  }
  if (threadIdx.x < M_ROWS) s.mrun[threadIdx.x] = -1e30f;
  __syncthreads();

  const int lo = chunk * per;
  const int hi = min(lo + per, kv_len);
  __half* srow = sbuf + (long)(chunk * H_KV + kvh) * M_ROWS * per;

  for (int t0 = lo; t0 < hi; t0 += BLOCK_A) {
    // stage K: 8 fp8 per thread per half-tile -> 8B load, one 16B vector store
#pragma unroll
    for (int ht = 0; ht < 2; ++ht) {
      const int c = threadIdx.x + ht * NTHREADS;      // chunk of 8 fp8 in the tile
      const int row = c / (D_HD / 8), col8 = (c % (D_HD / 8)) * 8;
      const int pid = idx[min(t0 / 16 + row / 16, npages - 1)];
      const uint8_t* p = kv + ((long)pid * 2) * (16 * H_KV * D_HD) +
                         (long)(row % 16) * (H_KV * D_HD) + kvh * D_HD + col8;
      const uint2 a2 = *reinterpret_cast<const uint2*>(p);
      __half o[8];
#pragma unroll
      for (int j = 0; j < 8; ++j)
        o[j] = e4m3_to_half(j < 4 ? ((a2.x >> (8 * j)) & 0xffu)
                                  : ((a2.y >> (8 * (j - 4))) & 0xffu));
      *reinterpret_cast<uint4*>(&s.k[row][col8]) = *reinterpret_cast<uint4*>(o);
    }
    __syncthreads();
    constexpr int NT = BLOCK_A / 16;
    if (warp < NMT * NT) {
      const int mt = warp / NT, nt = warp % NT;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
      wmma::fill_fragment(c, 0.f);
#pragma unroll
      for (int dc = 0; dc < D_HD / 16; ++dc) {
        wmma::load_matrix_sync(a, &s.q[mt * 16][dc * 16], D_HD);
        wmma::load_matrix_sync(b, &s.k[nt * 16][dc * 16], D_HD);
        wmma::mma_sync(c, a, b, c);
      }
      wmma::store_matrix_sync(&s.s[mt][0][nt * 16], c, BLOCK_A, wmma::mem_row_major);
    }
    __syncthreads();
    if (threadIdx.x < M_ROWS) {
      const int r = threadIdx.x;
      const int lim = kv_len - Q_LEN + (r / GROUP) + 1;
      const int mt = r / 16, lr = r % 16;
      float mx = s.mrun[r];
      __half* out = srow + (long)r * per + (t0 - lo);
#pragma unroll
      for (int j = 0; j < BLOCK_A; ++j) {
        const float x = (t0 + j < lim) ? s.s[mt][lr][j] * (1.0f / 16.0f) : -1e30f;
        out[j] = __float2half_rn(x);
        mx = fmaxf(mx, x);
      }
      s.mrun[r] = mx;
    }
    __syncthreads();
  }
  if (threadIdx.x < M_ROWS)
    mchunk[(long)(chunk * H_KV + kvh) * M_ROWS + threadIdx.x] = s.mrun[threadIdx.x];
}

// ---------------------------------------------------------------------------
// kernel B: softmax + P@V
// ---------------------------------------------------------------------------
struct SmemB {
  __half v[BLOCK_B][D_HD];     // 16 KB
  __half p[M_ROWS][BLOCK_B];   // 3 KB
};

extern "C" __global__ void __launch_bounds__(NTHREADS, 1)
mq_pv_kernel(const uint8_t* __restrict__ kv, const int* __restrict__ idx,
             const __half* __restrict__ sbuf, const float* __restrict__ mchunk,
             float* __restrict__ part_o, float* __restrict__ part_lse,
             int kv_len, int nchunks, int per, int npages) {
  extern __shared__ char raw[];
  SmemB& s = *reinterpret_cast<SmemB*>(raw);
  const int chunk = blockIdx.x, kvh = blockIdx.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;

  const int lo = chunk * per;
  const int hi = min(lo + per, kv_len);
  const __half* srow = sbuf + (long)(chunk * H_KV + kvh) * M_ROWS * per;
  const int nt = warp % (D_HD / 16);

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[NMT];
#pragma unroll
  for (int m = 0; m < NMT; ++m) wmma::fill_fragment(acc[m], 0.f);
  float lsum = 0.f;

  for (int t0 = lo; t0 < hi; t0 += BLOCK_B) {
    // P = exp(S - m) straight from global scores
    if (threadIdx.x < M_ROWS) {
      const int r = threadIdx.x;
      const float m = mchunk[(long)(chunk * H_KV + kvh) * M_ROWS + r];
      const __half* in = srow + (long)r * per + (t0 - lo);
      float ls = 0.f;
#pragma unroll
      for (int j = 0; j < BLOCK_B; ++j) {
        const float e = __expf(__half2float(in[j]) - m);
        s.p[r][j] = __float2half_rn(e);
        ls += e;
      }
      lsum += ls;
    }
    // stage V: 8 fp8 per thread per half-tile -> 8B load, one 16B vector store
#pragma unroll
    for (int ht = 0; ht < 2; ++ht) {
      const int c = threadIdx.x + ht * NTHREADS;
      const int row = c / (D_HD / 8), col8 = (c % (D_HD / 8)) * 8;
      const int pid = idx[min(t0 / 16 + row / 16, npages - 1)];
      const uint8_t* p = kv + ((long)pid * 2) * (16 * H_KV * D_HD) +
                         (long)(row % 16) * (H_KV * D_HD) + kvh * D_HD + col8 +
                         (long)16 * H_KV * D_HD;      // V plane
      const uint2 a2 = *reinterpret_cast<const uint2*>(p);
      __half o[8];
#pragma unroll
      for (int j = 0; j < 8; ++j)
        o[j] = e4m3_to_half(j < 4 ? ((a2.x >> (8 * j)) & 0xffu)
                                  : ((a2.y >> (8 * (j - 4))) & 0xffu));
      *reinterpret_cast<uint4*>(&s.v[row][col8]) = *reinterpret_cast<uint4*>(o);
    }
    __syncthreads();
    {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
#pragma unroll
      for (int m = 0; m < NMT; ++m)
#pragma unroll
        for (int kk = 0; kk < BLOCK_B / 16; ++kk) {   // BLOCK_B/16 k-steps
          wmma::load_matrix_sync(a, &s.p[m * 16][kk * 16], BLOCK_B);
          wmma::load_matrix_sync(b, &s.v[kk * 16][nt * 16], D_HD);
          wmma::mma_sync(acc[m], a, b, acc[m]);
        }
    }
    __syncthreads();
  }

  {
    float* o = part_o + ((long)(chunk * H_KV + kvh) * M_ROWS) * D_HD;
#pragma unroll
    for (int m = 0; m < NMT; ++m)
      wmma::store_matrix_sync(o + m * 16 * D_HD + nt * 16, acc[m], D_HD, wmma::mem_row_major);
  }
  if (threadIdx.x < M_ROWS)   // thread r owns row r's running sum
    part_lse[((long)(chunk * H_KV + kvh) * M_ROWS + threadIdx.x) * 2 + 1] = lsum;
  (void)lane;
}

// merge: out = sum_c O_c e^{m_c - m_g} / sum_c l_c e^{m_c - m_g}
extern "C" __global__ void __launch_bounds__(256)
mq_merge_kernel(const float* __restrict__ part_o, const float* __restrict__ part_lse,
                __half* __restrict__ out, int nchunks, int kvh) {
  const int r = blockIdx.x;
  if (r >= M_REAL) return;
  const int qr = r / GROUP, hig = r % GROUP;
  const int tid = threadIdx.x;
  float mg = -1e30f;
  for (int c = 0; c < nchunks; ++c)
    mg = fmaxf(mg, part_lse[((long)(c * H_KV + kvh) * M_ROWS + r) * 2]);
  float lg = 0.f;
  for (int c = 0; c < nchunks; ++c)
    lg += part_lse[((long)(c * H_KV + kvh) * M_ROWS + r) * 2 + 1] *
          __expf(part_lse[((long)(c * H_KV + kvh) * M_ROWS + r) * 2] - mg);
  for (int d = tid; d < D_HD; d += 256) {
    float a = 0.f;
    for (int c = 0; c < nchunks; ++c) {
      const float mc = part_lse[((long)(c * H_KV + kvh) * M_ROWS + r) * 2];
      a += part_o[((long)(c * H_KV + kvh) * M_ROWS + r) * D_HD + d] * __expf(mc - mg);
    }
    out[(qr * H_Q + kvh * GROUP + hig) * D_HD + d] = __float2half_rn(a / lg);
  }
}

// ---------------------------------------------------------------------------
// launchers
// ---------------------------------------------------------------------------
void mq_scores_launch(const void* q, const void* kv, const void* idx, void* sbuf,
                      void* mchunk, int kv_len, int nchunks, int per, int npages, void* stream) {
  const int smem = sizeof(SmemA);
  static int conf = -1;
  if (conf < smem) {
    conf = smem;
    cudaFuncSetAttribute(mq_scores_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  }
  dim3 grid(nchunks, H_KV);
  mq_scores_kernel<<<grid, NTHREADS, smem, (cudaStream_t)stream>>>(
      (const __half*)q, (const uint8_t*)kv, (const int*)idx, (__half*)sbuf,
      (float*)mchunk, kv_len, nchunks, per, npages);
}

void mq_pv_launch(const void* kv, const void* idx, const void* sbuf, const void* mchunk,
                  void* part_o, void* part_lse, int kv_len, int nchunks, int per, int npages,
                  void* stream) {
  const int smem = sizeof(SmemB);
  static int conf = -1;
  if (conf < smem) {
    conf = smem;
    cudaFuncSetAttribute(mq_pv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  }
  dim3 grid(nchunks, H_KV);
  mq_pv_kernel<<<grid, NTHREADS, smem, (cudaStream_t)stream>>>(
      (const uint8_t*)kv, (const int*)idx, (const __half*)sbuf, (const float*)mchunk,
      (float*)part_o, (float*)part_lse, kv_len, nchunks, per, npages);
}

void mq_merge_launch(const void* part_o, const void* part_lse, void* out, int nchunks,
                     void* stream) {
  for (int kvh = 0; kvh < H_KV; ++kvh)
    mq_merge_kernel<<<M_ROWS, 256, 0, (cudaStream_t)stream>>>(
        (const float*)part_o, (const float*)part_lse, (__half*)out, nchunks, kvh);
}

int mq_smem_a() { return (int)sizeof(SmemA); }
int mq_smem_b() { return (int)sizeof(SmemB); }
