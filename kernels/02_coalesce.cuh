#pragma once
#include <cuda_runtime.h>

// ============================================================================
// Kernel 2: 全局内存合并访问 (Global Memory Coalescing)
// ============================================================================
//
// 相比 naive,数学、工作量、线程总数「完全没变」,只改一件事:
//   让「同一个 warp 内相邻的线程」映射到「C 的相邻列」。
//
// 这样在固定的第 i 次迭代里,一个 warp 的 32 个线程:
//   - 读 B[i*N + cCol]:cCol = 0..31 连续 → 32 个连续 float → 合并成 1 次 128B 事务 ✓
//   - 读 A[cRow*K + i]:cRow 相同        → 同一地址          → 广播 ✓
//   - 写 C[cRow*N + cCol]:连续          → 合并 ✓
//
// 做法:改用「1D block」(BLOCKSIZE*BLOCKSIZE 个线程),手动把线程号切成行/列。
//   warp = threadIdx.x 连续 32 个,于是:
//     cRow = threadIdx.x / BLOCKSIZE   → 一个 warp 内不变
//     cCol = threadIdx.x % BLOCKSIZE   → 一个 warp 内 0..31 连续
// ----------------------------------------------------------------------------

template <const uint BLOCKSIZE>
__global__ void sgemm_coalesce(int M, int N, int K, float alpha,
                               const float *A, const float *B,
                               float beta, float *C) {
  const uint cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const uint cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (cRow < M && cCol < N) {
    float tmp = 0.0f;
    for (int i = 0; i < K; ++i) {
      tmp += A[cRow * K + i] * B[i * N + cCol];
    }
    C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
  }
}

void run_coalesce(int M, int N, int K, float alpha, float *A, float *B,
                  float beta, float *C) {
  const uint BLOCKSIZE = 32;
  dim3 gridDim(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));
  dim3 blockDim(BLOCKSIZE * BLOCKSIZE);  // 1D:1024 个线程
  sgemm_coalesce<BLOCKSIZE>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
