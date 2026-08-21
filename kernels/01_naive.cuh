#pragma once
#include <cuda_runtime.h>

// ============================================================================
// Kernel 1: Naive(朴素实现)
// ============================================================================
//
// 计算目标(SGEMM):  C = alpha * (A @ B) + beta * C
//   A: M×K   B: K×N   C: M×N   均为行主序(row-major)
//
// 并行策略:每个线程负责计算输出矩阵 C 中的「一个」元素。
//   线程 (row, col) 读取 A 的第 row 行 和 B 的第 col 列,
//   在 K 维上做点积,得到 C[row][col]。
//
// 线程之间完全独立,不需要 __syncthreads(),实现最简单。
// 但它非常慢(约为 cuBLAS 的 1.3%),原因见下方注释。
// ----------------------------------------------------------------------------

__global__ void sgemm_naive(int M, int N, int K, float alpha,
                            const float *A, const float *B,
                            float beta, float *C) {
  // ⚠️ 关键点:这里让 threadIdx.x(warp 内变化最快的维度)映射到「行 row」。
  //    一个 warp 里的 32 个线程 threadIdx.x = 0..31,对应 row = 0..31,而 col 相同。
  //    → 这些线程访问 C[row*N + col],地址相邻两个相差 N 个 float(跨了一整行),
  //      属于「非合并访问 uncoalesced」,是本 kernel 慢的主要原因。
  //    Kernel 2 只需把 row/col 的映射对调,就能大幅提速。
  const uint row = blockIdx.x * blockDim.x + threadIdx.x;
  const uint col = blockIdx.y * blockDim.y + threadIdx.y;

  // 边界检查:当 M/N 不是 blockDim 整数倍时,多出来的线程直接跳过
  if (row < M && col < N) {
    float tmp = 0.0f;
    // 点积:A 的一行 × B 的一列,共 K 次乘加(FMA)
    for (int i = 0; i < K; ++i) {
      tmp += A[row * K + i] * B[i * N + col];
    }
    // 写回,处理 alpha / beta
    C[row * N + col] = alpha * tmp + beta * C[row * N + col];
  }
}

// 主机端启动函数
void run_naive(int M, int N, int K, float alpha, float *A, float *B,
               float beta, float *C) {
  // 每个 block 32×32 = 1024 个线程(一个 block 计算 C 的一个 32×32 小块)
  dim3 blockDim(32, 32);
  // 网格覆盖整个 C:向上取整,保证边角也被覆盖(依赖 sgemm.cu 里的 CEIL_DIV 宏)
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
