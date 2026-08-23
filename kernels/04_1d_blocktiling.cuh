#pragma once
#include <cuda_runtime.h>

// ============================================================================
// Kernel 4: 1D Blocktiling(一维线程分块)
// ============================================================================
//
// 痛点:kernel 3 内层每次 FMA 要发 2 次 SMEM load(1 FMA : 2 load),
//       访存指令把计算单元堵住(MIO Throttle)。
//
// 思路:让「每个线程负责 TM=8 个输出」(C 里竖着的一小条)。
//       把点积循环 dotIdx 提到最外面,先把 Bs 的一个值读进寄存器 tmpB,
//       再用它连做 TM 次 FMA → 一次 SMEM load 服务多次计算,摊薄访存。
//       SMEM load : FMA 从 2:1 降到约 (TM+1):TM = 9:8。
//
// 分块尺寸:BM×BN = 64×64 输出块;BK=8 是 K 方向步长;TM=8 每线程输出数。
//   线程数 = BM*BN/TM = 64*64/8 = 512。
//   约束:BM*BK == 512 == blockDim,BN*BK == 512 == blockDim
//         → 每个线程恰好搬 1 个 As 元素 + 1 个 Bs 元素。
// ----------------------------------------------------------------------------

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_1d_blocktiling(int M, int N, int K, float alpha,
                                     const float *A, const float *B, float beta,
                                     float *C) {
  // 本 block 负责的输出块坐标(注意:cCol 用 x,cRow 用 y)
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  __shared__ float As[BM * BK];  // 64×8
  __shared__ float Bs[BK * BN];  // 8×64

  // 移动到本 block 负责的 A 行块 / B 列块 / C 目标块
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // ---- 两套索引:分清「加载布局」和「计算布局」----
  // 加载用:让连续 threadIdx.x 对应 tile 里连续的列 → GMEM 合并搬运
  const uint innerColA = threadIdx.x % BK;  // 0..7
  const uint innerRowA = threadIdx.x / BK;  // 0..63
  const uint innerColB = threadIdx.x % BN;  // 0..63
  const uint innerRowB = threadIdx.x / BN;  // 0..7
  // 计算用:512 个线程铺在 64×64 输出上,每线程占 1 列 × TM 行
  const uint threadCol = threadIdx.x % BN;  // 0..63
  const uint threadRow = threadIdx.x / BN;  // 0..7 → 负责第 threadRow*TM .. +TM-1 行

  // 每线程 TM 个结果,累加在寄存器里(寄存器是存储金字塔顶端,比 SMEM 还快)
  float threadResults[TM] = {0.0f};

  // 沿 K 维分块
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // 协作加载:每线程各搬 A、B 一个元素(512 线程刚好填满 512 元素的 tile)
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    A += BK;
    B += BK * N;

    // 计算:把 dotIdx 放最外层,方便复用 Bs 的值
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      float tmpB = Bs[dotIdx * BN + threadCol];  // 读 1 次,下面复用 TM 次
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  // 写回 TM 个结果(threadCol 连续 → 写 C 也是合并访问)
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        alpha * threadResults[resIdx] +
        beta * C[(threadRow * TM + resIdx) * N + threadCol];
  }
}

void run_1d_blocktiling(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  const uint BM = 64, BN = 64, BK = 8, TM = 8;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / TM);  // 512
  sgemm_1d_blocktiling<BM, BN, BK, TM>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
