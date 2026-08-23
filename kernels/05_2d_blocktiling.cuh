#pragma once
#include <cuda_runtime.h>

// ============================================================================
// Kernel 5: 2D Blocktiling(二维线程分块)
// ============================================================================
//
// 痛点:kernel 4 每线程算 TM=8 个输出(一竖条),每个 dotIdx 要
//       读 1 个 B + 8 个 A = 9 次 SMEM load,却只做 8 次 FMA(9:8)。
//       A 的值每个只用一次就扔,SMEM 流量还是太大。
//
// 思路:每线程负责 TM×TN = 8×8 = 64 个输出(一个小方块)。
//   每个 dotIdx:先把 TM 个 A 值 + TN 个 B 值搬进寄存器 tmpA/tmpB,
//   再用它们做 TM×TN = 64 次「外积」FMA:
//        SMEM load : FMA = (TM+TN) : TM*TN = 16 : 64 = 1:4
//   双重复用:tmpA[i] 被用 TN 次,tmpB[j] 被用 TM 次。
//
// 分块尺寸:BM=BN=128(输出块比 k4 翻倍),BK=8,TM=TN=8。
//   线程数 = BM*BN/(TM*TN) = 128*128/64 = 256。
//   加载:每线程搬 (BM*BK)/256 = 4 个 A + (BK*BN)/256 = 4 个 B
//   (k4「每线程恰好 1 个」的约束不再成立,改用小循环线性编号,
//    相邻线程 → 相邻线性位置 → 依然 GMEM 合并)。
//
// 代价:threadResults[64] + tmpA[8] + tmpB[8] ≈ 80+ 寄存器/线程,
//       寄存器压力上升 → 每 SM 驻留 block 数下降(occupancy 受限)。
// ----------------------------------------------------------------------------

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_2d_blocktiling(int M, int N, int K, float alpha,
                                     const float *A, const float *B,
                                     float beta, float *C) {
  // 本 block 负责的输出块坐标(行用 y、列用 x)
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  __shared__ float As[BM * BK];  // 128×8
  __shared__ float Bs[BK * BN];  // 8×128

  // 移动到本 block 负责的 A 行块 / B 列块 / C 目标块
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // ---- 计算布局:256 线程铺成 (BM/TM)×(BN/TN) = 16×16 的「线程网格」----
  // 每个线程负责输出块里一个 TM×TN = 8×8 的小方块
  const uint threadRow = threadIdx.x / (BN / TN);  // 0..15
  const uint threadCol = threadIdx.x % (BN / TN);  // 0..15

  // 常量:线程数、每线程加载的元素个数(编译期即可算出)
  const uint numThreads = (BM * BN) / (TM * TN);        // 256
  const uint A_LOADS = (BM * BK) / numThreads;          // 4
  const uint B_LOADS = (BK * BN) / numThreads;          // 4

  // 64 个累加器,全程住在寄存器里(这正是本 kernel 快的原因)
  float threadResults[TM * TN] = {0.0f};

  // 沿 K 维分块
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // ---- 协作加载:每线程 4 个 A + 4 个 B ----
    // tile 线性编号:线程 i 负责线性位置 i, i+256, i+512, i+768。
    // 相邻线程 → 相邻线性位置 → GMEM 地址连续(合并)。
    for (uint loadIdx = 0; loadIdx < A_LOADS; ++loadIdx) {
      const uint linear = loadIdx * numThreads + threadIdx.x;
      // As 是 BM×BK 行主序,linear 直接就是 SMEM 下标;
      // 全局地址:行 = linear/BK,列 = linear%BK
      As[linear] = A[(linear / BK) * K + (linear % BK)];
    }
    for (uint loadIdx = 0; loadIdx < B_LOADS; ++loadIdx) {
      const uint linear = loadIdx * numThreads + threadIdx.x;
      // Bs 是 BK×BN 行主序;全局地址:行 = linear/BN,列 = linear%BN
      Bs[linear] = B[(linear / BN) * N + (linear % BN)];
    }
    __syncthreads();  // 数据齐了才能算

    A += BK;
    B += BK * N;

    // ---- 计算:先搬 16 个值进寄存器,再做 64 次外积 FMA ----
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      float tmpA[TM], tmpB[TN];
#pragma unroll
      for (uint i = 0; i < TM; ++i)
        tmpA[i] = As[(threadRow * TM + i) * BK + dotIdx];  // 本方块的 A 列段
#pragma unroll
      for (uint j = 0; j < TN; ++j)
        tmpB[j] = Bs[dotIdx * BN + threadCol * TN + j];    // 本方块的 B 行段

#pragma unroll
      for (uint i = 0; i < TM; ++i)
#pragma unroll
        for (uint j = 0; j < TN; ++j)
          threadResults[i * TN + j] += tmpA[i] * tmpB[j];
    }
    __syncthreads();  // 算完了才能覆盖 SMEM
  }

  // ---- 写回 8×8 小方块 ----
  // 同一 warp 内 threadCol 连续 → 内层 j 连续 → 写 C 合并访问
  for (uint i = 0; i < TM; ++i)
    for (uint j = 0; j < TN; ++j) {
      const uint row = threadRow * TM + i;
      const uint col = threadCol * TN + j;
      C[row * N + col] =
          alpha * threadResults[i * TN + j] + beta * C[row * N + col];
    }
}

void run_2d_blocktiling(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  const uint BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));  // x 覆盖列,y 覆盖行
  dim3 blockDim((BM * BN) / (TM * TN));            // 256
  sgemm_2d_blocktiling<BM, BN, BK, TM, TN>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
