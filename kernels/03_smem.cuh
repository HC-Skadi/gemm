#pragma once
#include <cuda_runtime.h>

// ============================================================================
// Kernel 3: 共享内存缓存分块 (Shared Memory Cache-Blocking)
// ============================================================================
//
// 痛点:kernel 2 里,A 的每一行要被 N 个线程反复从「慢速全局内存」各读一遍,
//       B 的每一列被 M 个线程各读一遍。带宽全浪费在重复读上。
//
// 思路:一个 block 内的线程本来就共用同一片 A/B 数据,那就先把这片 tile 从
//       全局内存搬进「片上共享内存 SMEM」(快约 100 倍),大家从 SMEM 反复读。
//
// 做法(沿 K 维分块):
//   一个 block 负责 C 的一个 BS×BS 输出块。把 K 切成若干段,每段:
//     1) 每个线程协作搬运 A/B 各 1 个元素进 SMEM(搬运时保持合并访问)
//     2) __syncthreads():等全 block 搬完
//     3) 从 SMEM 做这一段的部分点积,累加到寄存器 tmp
//     4) __syncthreads():等全 block 算完,才能覆盖 SMEM 进入下一段
//   每个 SMEM 元素会被 BS 个线程复用 → 全局内存流量降为原来的 1/BS。
// ----------------------------------------------------------------------------

template <const int BS>  // BS = BLOCKSIZE = 分块边长(同时也是 K 方向步长 BK)
__global__ void sgemm_smem(int M, int N, int K, float alpha, const float *A,
                           const float *B, float beta, float *C) {
  // 本 block 负责的输出块坐标(以「块」为单位)
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // 片上共享内存:各缓存一个 BS×BS 的 tile
  __shared__ float As[BS * BS];
  __shared__ float Bs[BS * BS];

  // 线程在块内的 (行, 列);让 threadCol = threadIdx.x % BS 连续 → 搬运时合并
  const uint threadRow = threadIdx.x / BS;
  const uint threadCol = threadIdx.x % BS;

  // 把指针移到本 block 负责的起始位置
  A += cRow * BS * K;              // A 的第 cRow 个行块,列从 0 开始
  B += cCol * BS;                  // B 的第 cCol 个列块,行从 0 开始
  C += cRow * BS * N + cCol * BS;  // C 的目标块

  float tmp = 0.0f;
  // 沿 K 维一段一段推进
  for (int bkIdx = 0; bkIdx < K; bkIdx += BS) {
    // 1) 协作加载:每个线程搬 A、B 各一个元素。threadCol 连续 → 合并访问 ✓
    As[threadRow * BS + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BS + threadCol] = B[threadRow * N + threadCol];

    // 2) 等所有线程都把这一段搬进 SMEM
    __syncthreads();

    // 推进全局指针到下一段
    A += BS;
    B += BS * N;

    // 3) 用 SMEM 里的这一段做部分点积(全部命中快速 SMEM)
    for (int dotIdx = 0; dotIdx < BS; ++dotIdx) {
      tmp += As[threadRow * BS + dotIdx] * Bs[dotIdx * BS + threadCol];
    }

    // 4) 等所有线程算完,才能在下一轮覆盖 SMEM
    __syncthreads();
  }

  C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];
}

void run_smem(int M, int N, int K, float alpha, float *A, float *B, float beta,
              float *C) {
  const uint BS = 32;
  dim3 gridDim(CEIL_DIV(M, BS), CEIL_DIV(N, BS));
  dim3 blockDim(BS * BS);  // 1D:1024 个线程,一个 block 算 32×32 输出块
  sgemm_smem<BS><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
