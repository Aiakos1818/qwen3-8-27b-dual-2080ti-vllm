// Multi-query (spec-verify shaped) paged attention for SM75, fp8_e4m3 KV.
//
// Why this exists: flashinfer's batch-prefill kernel is the only available
// implementation for q_len > 1 on Turing, and it runs with one 64 KB smem CTA
// per SM and four hardcoded warps, i.e. 12.5% occupancy.  Measured 1.59 ms and
// 161 GB/s for a 250K-token, 12-qo/2-kv-head, head_dim-256, fp8-KV verify
// batch, while a plain streaming kernel touches the same paged fp8 bytes at
// 542 GB/s.
//
// Design notes
//   * 36 = q_len(6) x group(6) (row, head) pairs are packed into a 48-row M
//     tile (three 16-row wmma tiles); the KV is read once per kv head and
//     shared by all six qo heads of that head.
//   * Two passes over the KV *chunk*: pass A reads only K to get each row's
//     softmax max and sum, pass B recomputes the scores and does P@V with the
//     final max.  Two passes cost 1.5x the KV traffic but avoid touching the
//     wmma accumulator element-by-element, whose per-lane layout is opaque on
//     this architecture; only store_matrix_sync / load_matrix_sync are used.
//   * One page (16 keys) per tile, so the paged gather is one index lookup per
//     tile and every staged row is page-aligned.
//   * K/V are staged as fp16 (Turing has no fp8 convert instruction, so the
//     conversion is plain bit manipulation) and consumed by wmma.
//
// Written for: sm_75, by a wide margin the weakest supported target.

#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>

using namespace nvcuda;

#define D_HD 256      // head_dim (qk and vo)
#define Q_LEN 6       // queries per request (1 + num_speculative_tokens)
#define GROUP 6       // qo heads sharing a kv head
#define H_KV 2        // kv heads per rank
#define H_Q 12        // qo heads per rank
#define M_REAL (Q_LEN * GROUP)          // 36
#define M_ROWS 48                       // 3 x 16
#define N_MTILE 3
#define TILE 16                         // keys per iteration == page size
#define NTHREADS 512
#define NWARPS (NTHREADS / 32)

__device__ __forceinline__ __half e4m3_to_half(unsigned u) {
  // normal path is pure bit surgery; exp==0 holds only 7 subnormal values
  unsigned s = (u >> 7) & 1u, e = (u >> 3) & 0xFu, m = u & 0x7u;
  unsigned bits = (s << 15) | ((e + 8u) << 10) | (m << 7);
  __half hn;
  memcpy(&hn, &bits, 2);
  __half hs = __float2half_rn((float)m * (1.0f / 512.0f));
  return e == 0u ? hs : hn;
}

struct Smem {
  __half q[M_ROWS][D_HD];      // 24 KB
  __half k[TILE][D_HD];        // 8 KB
  __half v[TILE][D_HD];        // 8 KB
  __half p[M_ROWS][TILE];      // 1.5 KB
  float s[N_MTILE][16][16];    // 3 KB  (S tiles, fp32)
  float mrun[M_ROWS];
  float lrun[M_ROWS];
};

// ---------------------------------------------------------------------------
// pass A: K only, running (max, sum) per row
// ---------------------------------------------------------------------------
__device__ __forceinline__ void stage_k(Smem& s, const uint8_t* __restrict__ kv,
                                        const int* __restrict__ idx, int t0,
                                        int kvh) {
  const int i = threadIdx.x * 8;           // 512 threads x 8 fp8 = 4096 = TILE*D
  const int ki = i / D_HD, d = i % D_HD;
  const int pid = idx[t0 / TILE];
  const uint8_t* kp = kv + ((long)pid * 2 * TILE) * (H_KV * (long)D_HD) +
                      (long)ki * (H_KV * D_HD) + kvh * D_HD + d;
  const uint2 k2 = *reinterpret_cast<const uint2*>(kp);
  const uint64_t u = ((uint64_t)k2.y << 32) | k2.x;
#pragma unroll
  for (int j = 0; j < 8; ++j)
    s.k[ki][d + j] = e4m3_to_half((unsigned)((u >> (8 * j)) & 0xffu));
}

template <bool ENABLE_CAUSAL>
__device__ __forceinline__ void qk_tile(Smem& s, int warp) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
  wmma::fill_fragment(c, 0.f);
#pragma unroll
  for (int dc = 0; dc < D_HD / 16; ++dc) {
    wmma::load_matrix_sync(a, &s.q[warp * 16][dc * 16], D_HD);
    wmma::load_matrix_sync(b, &s.k[0][dc * 16], D_HD);   // K^T, ld = D_HD
    wmma::mma_sync(c, a, b, c);
  }
  wmma::store_matrix_sync(&s.s[warp][0][0], c, 16, wmma::mem_row_major);
}

__device__ __forceinline__ void mask_and_softmax_stats(
    Smem& s, int t0, int kv_len, int nchunks, int chunk) {
  // 48 rows: thread r handles row r
  const int r = threadIdx.x;
  if (r >= M_ROWS) return;
  const int qr = r % Q_LEN;
  const int lim = kv_len - Q_LEN + qr + 1;      // exclusive key limit
  const int row = r / 16, lr = r % 16;
  float* sr = s.s[row][lr];
  float tmax = -1e30f;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const int k = t0 + j;
    // scale = 1/sqrt(256) = 1/16; masked keys never win the max
    sr[j] = (k < lim) ? sr[j] * (1.0f / 16.0f) : -1e30f;
    tmax = fmaxf(tmax, sr[j]);
  }
  const float m_old = s.mrun[r];
  const float m_new = fmaxf(m_old, tmax);
  float sum = 0.f;
#pragma unroll
  for (int j = 0; j < 16; ++j) sum += __expf(sr[j] - m_new);
  s.lrun[r] = s.lrun[r] * __expf(m_old - m_new) + sum;
  s.mrun[r] = m_new;
}

// ---------------------------------------------------------------------------
// pass B: K + V, recompute S, P = exp(S - m_final), accumulate P@V
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint64_t load_raw(const uint8_t* __restrict__ kv,
                                            const int* __restrict__ idx, int t0,
                                            int kvh, int plane_off) {
  const int i = threadIdx.x * 8;
  const int ki = i / D_HD, d = i % D_HD;
  const int pid = idx[t0 / TILE];
  const uint8_t* p = kv + ((long)pid * 2 * TILE) * (H_KV * (long)D_HD) +
                     (long)ki * (H_KV * D_HD) + kvh * D_HD + d + plane_off;
  const uint2 v = *reinterpret_cast<const uint2*>(p);   // 8B aligned by construction
  return ((uint64_t)v.y << 32) | v.x;
}

__device__ __forceinline__ void stage_kv(Smem& s, const uint8_t* __restrict__ kv,
                                         const int* __restrict__ idx, int t0,
                                         int kvh) {
  const int i = threadIdx.x * 8;
  const int ki = i / D_HD, d = i % D_HD;
  const int pid = idx[t0 / TILE];
  const uint8_t* kp = kv + ((long)pid * 2 * TILE) * (H_KV * (long)D_HD) +
                      (long)ki * (H_KV * D_HD) + kvh * D_HD + d;
  const uint2 k2 = *reinterpret_cast<const uint2*>(kp);  // 8B aligned: strides are x8
  const uint2 v2 = *reinterpret_cast<const uint2*>(kp + (long)TILE * H_KV * D_HD);
  const uint64_t ku = ((uint64_t)k2.y << 32) | k2.x;
  const uint64_t vu = ((uint64_t)v2.y << 32) | v2.x;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    s.k[ki][d + j] = e4m3_to_half((unsigned)((ku >> (8 * j)) & 0xffu));
    s.v[ki][d + j] = e4m3_to_half((unsigned)((vu >> (8 * j)) & 0xffu));
  }
}

__device__ __forceinline__ void p_from_s(Smem& s, int t0, int kv_len) {
  for (int i = threadIdx.x; i < M_ROWS * TILE; i += NTHREADS) {
    const int r = i / TILE, j = i % TILE;
    const int qr = r % Q_LEN;
    const int lim = kv_len - Q_LEN + qr + 1;
    const int k = t0 + j;
    const float x = s.s[r / 16][r % 16][j] * (1.0f / 16.0f);
    s.p[r][j] = (k < lim) ? __float2half_rn(__expf(x - s.mrun[r]))
                          : __float2half_rn(0.f);
  }
}

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(NTHREADS, 1)
mq_attn_kernel(const __half* __restrict__ q, const uint8_t* __restrict__ kv,
               const int* __restrict__ idx, float* __restrict__ part_o,
               float* __restrict__ part_lse, int kv_len, int nchunks, int per,
               int mode) {
  extern __shared__ char smem_raw[];
  Smem& s = *reinterpret_cast<Smem*>(smem_raw);
  if (mode == 8) {                       // empty kernel: isolate fixed overhead
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) part_o[0] = 0.f;
    return;
  }
  const int chunk = blockIdx.x;
  const int kvh = blockIdx.y;
  const int warp = threadIdx.x >> 5;

  // Q into smem: row r = (q_row * GROUP + head_in_group) keeps the causal
  // limit and the head mapping computable from r alone.
  for (int i = threadIdx.x; i < M_ROWS * D_HD; i += NTHREADS) {
    const int r = i / D_HD, d = i % D_HD;
    __half val = __float2half_rn(0.f);
    if (r < M_REAL) {
      const int qr = r / GROUP, hig = r % GROUP;
      val = q[(qr * H_Q + kvh * GROUP + hig) * D_HD + d];
    }
    s.q[r][d] = val;
  }
  if (threadIdx.x < M_ROWS) {
    s.mrun[threadIdx.x] = -1e30f;
    s.lrun[threadIdx.x] = 0.f;
  }
  __syncthreads();

  const int lo = chunk * per;
  const int hi = min(lo + per, kv_len);

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[N_MTILE];

  // ---- pass A: running max / sum ----
  for (int t0 = lo; t0 < hi && mode != 3; t0 += TILE) {
    stage_k(s, kv, idx, t0, kvh);
    __syncthreads();
    if (warp < N_MTILE) qk_tile<true>(s, warp);
    __syncthreads();
    if (mode != 2) mask_and_softmax_stats(s, t0, kv_len, nchunks, chunk);
    __syncthreads();
  }

  // ---- pass B: P@V with the final max, no accumulator rescaling ----
  for (int m = 0; m < N_MTILE; ++m) wmma::fill_fragment(acc[m], 0.f);
  if (mode == 6) {   // probe pattern: row-strided, uint4, no smem / no barriers
    const int vecs = D_HD / 16, rows_per_iter = NTHREADS / vecs;
    uint64_t acc = 0;
#pragma unroll 4
    for (int row = lo + threadIdx.x / vecs; row < hi; row += rows_per_iter) {
      const int col = (threadIdx.x % vecs) * 16;
      const int pid = idx[row / TILE];
      const uint8_t* p = kv + ((long)pid * 2 * TILE) * (H_KV * (long)D_HD) +
                         (long)(row % TILE) * (H_KV * D_HD) + kvh * D_HD + col;
      const uint4 k4 = *reinterpret_cast<const uint4*>(p);
      const uint4 v4 = *reinterpret_cast<const uint4*>(p + (long)TILE * H_KV * D_HD);
      acc += k4.x + k4.y + k4.z + k4.w + v4.x + v4.y + v4.z + v4.w;
    }
    if (acc == 12345678ull) part_o[0] = 2.f;
  }
  uint64_t dummy = 0;
  if (mode == 7) {   // like mode 6 but with register double-buffering (prefetch)
    const int vecs = D_HD / 16, rpi = NTHREADS / vecs;
    const int col = (threadIdx.x % vecs) * 16;
    int row = lo + threadIdx.x / vecs;
    uint4 k4, v4;
    if (row < hi) {
      const int p0 = idx[row / TILE];
      const uint8_t* pp = kv + ((long)p0 * 2 * TILE) * (H_KV * (long)D_HD) +
                          (long)(row % TILE) * (H_KV * D_HD) + kvh * D_HD + col;
      k4 = *reinterpret_cast<const uint4*>(pp);
      v4 = *reinterpret_cast<const uint4*>(pp + (long)TILE * H_KV * D_HD);
    }
    for (; row < hi; row += rpi) {
      const int nr = row + rpi;
      uint4 k5 = k4, v5 = v4;
      if (nr < hi) {   // issue the next tile's loads before consuming this one
        const int p1 = idx[nr / TILE];
        const uint8_t* qq = kv + ((long)p1 * 2 * TILE) * (H_KV * (long)D_HD) +
                            (long)(nr % TILE) * (H_KV * D_HD) + kvh * D_HD + col;
        k5 = *reinterpret_cast<const uint4*>(qq);
        v5 = *reinterpret_cast<const uint4*>(qq + (long)TILE * H_KV * D_HD);
      }
      dummy += k4.x + k4.y + k4.z + k4.w + v4.x + v4.y + v4.z + v4.w;
      k4 = k5; v4 = v5;
    }
    if (dummy == 12345678ull) part_o[0] = 3.f;
  }
  for (int t0 = lo; t0 < hi && mode != 6 && mode != 7; t0 += TILE) {
    if (mode == 4) { dummy += load_raw(kv, idx, t0, kvh, 0) + load_raw(kv, idx, t0, kvh, TILE*H_KV*D_HD); continue; }
    if (mode == 5) {
      uint64_t ku = load_raw(kv, idx, t0, kvh, 0), vu = load_raw(kv, idx, t0, kvh, TILE*H_KV*D_HD);
      const int i = threadIdx.x * 8; const int ki = i / D_HD, d = i % D_HD;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        s.k[ki][d+j] = e4m3_to_half((unsigned)((ku >> (8*j)) & 0xff));
        s.v[ki][d+j] = e4m3_to_half((unsigned)((vu >> (8*j)) & 0xff));
      }
      continue;
    }
    stage_kv(s, kv, idx, t0, kvh);
    __syncthreads();
    if (warp < N_MTILE) qk_tile<true>(s, warp);
    __syncthreads();
    if (mode != 2) p_from_s(s, t0, kv_len);
    __syncthreads();
    // every warp owns n-tile `warp % 16` (only 16 n-tiles of 16 in D_HD=256)
    {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
      const int nt = warp % (D_HD / 16);
#pragma unroll
      for (int m = 0; m < N_MTILE; ++m) {
        wmma::load_matrix_sync(a, &s.p[m * 16][0], TILE);
        wmma::load_matrix_sync(b, &s.v[0][nt * 16], D_HD);
        wmma::mma_sync(acc[m], a, b, acc[m]);
      }
    }
    __syncthreads();
  }

  // ---- write partials ----
  {
    const int nt = warp % (D_HD / 16);
#pragma unroll
    for (int m = 0; m < N_MTILE; ++m) {
      wmma::store_matrix_sync(&part_o[((long)(chunk * H_KV + kvh) * M_ROWS + m * 16) * D_HD +
                                      nt * 16],
                              acc[m], D_HD, wmma::mem_row_major);
    }
  }
  if (mode == 4 && dummy == 12345678ull) part_o[0] = 1.f;
  if (threadIdx.x < M_ROWS) {
    long b = ((long)(chunk * H_KV + kvh) * M_ROWS + threadIdx.x) * 2;
    part_lse[b] = s.mrun[threadIdx.x];
    part_lse[b + 1] = s.lrun[threadIdx.x];
  }
}

// merge the per-chunk partials: out = sum_c O_c e^{m_c - m_g} / sum_c l_c e^{m_c - m_g}
extern "C" __global__ void __launch_bounds__(256)
mq_merge_kernel(const float* __restrict__ part_o, const float* __restrict__ part_lse,
                __half* __restrict__ out, int nchunks, int kvh) {
  const int r = blockIdx.x;                 // 0..M_ROWS-1
  if (r >= M_REAL) return;
  const int qr = r / GROUP, hig = r % GROUP;
  const int tid = threadIdx.x;

  float mg = -1e30f;
  for (int c = 0; c < nchunks; ++c)
    mg = fmaxf(mg, part_lse[((long)(c * H_KV + kvh) * M_ROWS + r) * 2]);
  float lg = 0.f;
  for (int c = 0; c < nchunks; ++c) {
    const float mc = part_lse[((long)(c * H_KV + kvh) * M_ROWS + r) * 2];
    const float lc = part_lse[((long)(c * H_KV + kvh) * M_ROWS + r) * 2 + 1];
    lg += lc * __expf(mc - mg);
  }
  const long obase = ((long)(kvh)*M_ROWS + r) * D_HD;
  for (int d = tid; d < D_HD; d += 256) {
    float a = 0.f;
    for (int c = 0; c < nchunks; ++c) {
      const float mc = part_lse[((long)(c * H_KV + kvh) * M_ROWS + r) * 2];
      a += part_o[(obase + (long)c * H_KV * M_ROWS * D_HD) + d] * __expf(mc - mg);
    }
    out[(qr * H_Q + kvh * GROUP + hig) * D_HD + d] = __float2half_rn(a / lg);
  }
}

// ---------------------------------------------------------------------------
// host wrappers
// ---------------------------------------------------------------------------
int g_dbg[4] = {0, 0, 0, 0};

void mq_attn_launch_raw(const void* q, const void* kv, const void* idx, void* part_o,
                        void* part_lse, int kv_len, int nchunks, int per,
                        int mode, int smem_override, void* stream) {
  const int smem = sizeof(Smem);
  const int smem_launch = (smem_override > 0) ? smem_override : smem;
  // cudaFuncSetAttribute is a synchronous runtime call: do it once, not per launch.
  static int configured = -1;
  if (configured < smem_launch) {
    configured = smem_launch;
    cudaError_t e = cudaFuncSetAttribute(mq_attn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_launch);
    g_dbg[0] = (int)e;
  }
  dim3 grid(nchunks, H_KV);
  // no stream on purpose: isolate stream-vs-smem as the cause
  mq_attn_kernel<<<grid, NTHREADS, smem_launch>>>(
      (const __half*)q, (const uint8_t*)kv, (const int*)idx, (float*)part_o,
      (float*)part_lse, kv_len, nchunks, per, mode);
  g_dbg[2] = (int)cudaGetLastError();
  g_dbg[3] = smem_launch;
}

int mq_dbg(int i) { return (i >= 0 && i < 4) ? g_dbg[i] : -1; }

void mq_merge_launch_raw(const void* part_o, const void* part_lse, void* out,
                         int nchunks, void* stream) {
  for (int kvh = 0; kvh < H_KV; ++kvh)
    mq_merge_kernel<<<M_ROWS, 256, 0, (cudaStream_t)stream>>>(
        (const float*)part_o, (const float*)part_lse, (__half*)out, nchunks, kvh);
}

int mq_smem_bytes() { return (int)sizeof(Smem); }

int mq_try_setattr(int bytes) {
  cudaError_t e = cudaFuncSetAttribute(mq_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  return (int)e;
}
int mq_try_launch(int bytes) {
  dim3 grid(1, 1);
  extern __shared__ char dummy[];
  mq_attn_kernel<<<grid, NTHREADS, bytes>>>(
      (const __half*)nullptr, (const uint8_t*)nullptr, (const int*)nullptr,
      (float*)nullptr, (float*)nullptr, 16, 1, 16, 0);
  return (int)cudaGetLastError();
}
int mq_merge_err() {
  mq_merge_kernel<<<1, 256>>>(nullptr, nullptr, nullptr, 1, 0);
  return (int)cudaGetLastError();
}

int mq_dbg(int);

const char* mq_last_error() {
  cudaError_t e = cudaGetLastError();
  return cudaGetErrorString(e);
}

__device__ __forceinline__ int tid_add(int a, int b, int c) { return a ^ b ^ c; }

// Diagnostic: the *exact* read loop of mode 6, alone in its own kernel.
extern "C" __global__ void __launch_bounds__(512, 1)
mq_readonly_kernel(const uint8_t* __restrict__ kv, const int* __restrict__ idx,
                   int kv_len, int nchunks, int per, int kvh, float* __restrict__ sink) {
  const int chunk = blockIdx.x;
  const int lo = chunk * per;
  const int hi = min(lo + per, kv_len);
  if (lo >= kv_len) return;
  const int vecs = D_HD / 16, rpi = NTHREADS / vecs;
  const int col = (threadIdx.x % vecs) * 16;
  uint64_t acc = 0;
  for (int row = lo + threadIdx.x / vecs; row < hi; row += rpi) {
    const int pid = idx[row / TILE];
    const uint8_t* p = kv + ((long)pid * 2 * TILE) * (H_KV * (long)D_HD) +
                       (long)(row % TILE) * (H_KV * D_HD) + kvh * D_HD + col;
    const uint4 k4 = *reinterpret_cast<const uint4*>(p);
    const uint4 v4 = *reinterpret_cast<const uint4*>(p + (long)TILE * H_KV * D_HD);
    acc += k4.x + k4.y + k4.z + k4.w + v4.x + v4.y + v4.z + v4.w;
  }
  if (acc == 12345678ull) sink[chunk] = 1.f;
}

void mq_readonly_launch(const void* kv, const void* idx, int kv_len, int nchunks,
                        int per, void* sink, void* stream) {
  dim3 grid(nchunks, H_KV);
  mq_readonly_kernel<<<grid, NTHREADS, 0, (cudaStream_t)stream>>>(
      (const uint8_t*)kv, (const int*)idx, kv_len, nchunks, per, 0, (float*)sink);
}
