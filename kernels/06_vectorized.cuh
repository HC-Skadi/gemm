#pragma once
#include <cuda_runtime.h>

// ============================================================================
// Kernel 6: 向量化访存(Vectorized ld/st,float4)
// ============================================================================
//
// 痛点:k5 的加载阶段,每线程每个 BK 步要发 8 条 32 位 load 指令
//       (4 个 A + 4 个 B),SMEM 端同样逐元素写入,指令条数多;
//       32 位传输也用不满内存通路。
//
// 思路:GMEM→SMEM 的搬运改用 float4(128 位)指令,一条指令搬 4 个 float。
//   本参数下尺寸恰好对齐:
//     A tile: BM*(BK/4) = 128*2 = 256 个 float4(BK/4=2 个/行 × 128 行)
//     B tile: BK*(BN/4) = 8*32  = 256 个 float4(BN/4=32 个/行 × 8 行)
//   都等于线程数 256 → 每线程恰好 1 条 LDG.128 搬 A、1 条搬 B。
//   加载指令从 8 条 LDG.32 降到 2 条 LDG.128(1/4)。
//
// 对齐前提(不满足会得到错误结果):
//   - GMEM:K、N 为 4 的倍数(行首 16B 对齐);cudaMalloc 基址 256B 对齐
//   - SMEM:数组声明 __align__(16);行跨度 BK=8 float=32B、
//     BN=128 float=512B,均为 16 的倍数 → 行内每个 float4 都对齐
//   - 约束:BM*BK/4 == BK*BN/4 == 线程数(改参数时要注意)
//
// 计算内层与 k5 相同(tmpA 沿 BK 跨步无法向量化,tmpB 可以但收益小,
// 本课聚焦 GMEM→SMEM 路径,保持不变)。
// ----------------------------------------------------------------------------

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_vectorized(int M, int N, int K, float alpha,
                                 const float *A, const float *B, float beta,
                                 float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  __shared__ __align__(16) float As[BM * BK];
  __shared__ __align__(16) float Bs[BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // ---- 计算布局:与 k5 相同 ----
  const uint threadRow = threadIdx.x / (BN / TN);  // 0..15
  const uint threadCol = threadIdx.x % (BN / TN);  // 0..15

  // ---- 加载布局:每线程 1 个 A 向量 + 1 个 B 向量 ----
  // 向量坐标在整个 K 循环里不变,div/mod 只算一次
  // (对比 k5:加载的除法/取模在循环内每个 BK 步都要重算)
  const uint aVecRow = threadIdx.x / (BK / 4);  // 0..127(A tile 的行)
  const uint aVecCol = threadIdx.x % (BK / 4);  // 0..1(行内第几个 float4)
  const uint bVecRow = threadIdx.x / (BN / 4);  // 0..7(B tile 的行)
  const uint bVecCol = threadIdx.x % (BN / 4);  // 0..31

  float threadResults[TM * TN] = {0.0f};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // GMEM→SMEM:一条 LDG.128 搬 4 个 float,SMEM 端也是 128 位写入
    reinterpret_cast<float4 *>(&As[aVecRow * BK])[aVecCol] =
        reinterpret_cast<const float4 *>(&A[aVecRow * K])[aVecCol];
    reinterpret_cast<float4 *>(&Bs[bVecRow * BN])[bVecCol] =
        reinterpret_cast<const float4 *>(&B[bVecRow * N])[bVecCol];
    __syncthreads();  // 数据齐了才能算

    A += BK;
    B += BK * N;

    // ---- 计算:与 k5 完全相同 ----
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      float tmpA[TM], tmpB[TN];
#pragma unroll
      for (uint i = 0; i < TM; ++i)
        tmpA[i] = As[(threadRow * TM + i) * BK + dotIdx];
#pragma unroll
      for (uint j = 0; j < TN; ++j)
        tmpB[j] = Bs[dotIdx * BN + threadCol * TN + j];

#pragma unroll
      for (uint i = 0; i < TM; ++i)
#pragma unroll
        for (uint j = 0; j < TN; ++j)
          threadResults[i * TN + j] += tmpA[i] * tmpB[j];
    }
    __syncthreads();  // 算完了才能覆盖 SMEM
  }

  // ---- 写回:与 k5 相同(TN=8 个连续列,同样可以改成 2 条 float4 写,
  //      属于同一手法的重复应用,留作练习)----
  for (uint i = 0; i < TM; ++i)
    for (uint j = 0; j < TN; ++j) {
      const uint row = threadRow * TM + i;
      const uint col = threadCol * TN + j;
      C[row * N + col] =
          alpha * threadResults[i * TN + j] + beta * C[row * N + col];
    }
}

void run_vectorized(int M, int N, int K, float alpha, float *A, float *B,
                    float beta, float *C) {
  const uint BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / (TM * TN));  // 256
  sgemm_vectorized<BM, BN, BK, TM, TN>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
