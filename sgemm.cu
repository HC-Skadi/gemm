// ============================================================================
// SGEMM 学习项目 —— 主驱动 / host harness
//
// 参考文章: https://siboehm.com/articles/22/CUDA-MMM
//
// 职责:
//   1. 分配 & 随机初始化矩阵
//   2. 用 cuBLAS 算出「参考结果」并作为性能基线(kernel 0)
//   3. 运行被测 kernel,校验正确性
//   4. 用 cudaEvent 计时,换算成 GFLOPs/s
//
// 用法:
//   ./sgemm <kernel_num> [size] [repeats]
//     kernel_num : 0=cuBLAS  1=naive  ...(后续课程会不断添加)
//     size       : 方阵边长 M=N=K,默认 4096
//     repeats    : 计时重复次数,默认 50(naive 很慢,建议 ./sgemm 1 2048 10)
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// 向上取整除法:用于计算需要多少个 block
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// CUDA API 错误检查
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,                     \
             cudaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ---- 各个 kernel(逐课添加)-------------------------------------------------
#include "kernels/01_naive.cuh"
#include "kernels/02_coalesce.cuh"
#include "kernels/03_smem.cuh"
#include "kernels/04_1d_blocktiling.cuh"
#include "kernels/05_2d_blocktiling.cuh"
#include "kernels/06_vectorized.cuh"
#include "kernels/09_autotuned.cuh"
#include "kernels/10_warptiling.cuh"

constexpr int WARPSIZE = 32;

// ---- cuBLAS 参考实现(同时是性能基线 kernel 0)----------------------------
//
// cuBLAS 是「列主序」,而我们的数据是「行主序」。
// 数学恒等式:行主序的 C = A·B  等价于  列主序的 C^T = B^T · A^T。
// 因此只要把 B、A 交换顺序、并把维度 M/N 对调传进去,
// cuBLAS 写出来的内存布局刚好就是我们想要的行主序 C,无需任何转置拷贝。
void run_cublas(cublasHandle_t handle, int M, int N, int K, float alpha,
                float *A, float *B, float beta, float *C) {
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K,
              &beta, C, N);
}

// ---- kernel 分发 -----------------------------------------------------------
void run_kernel(int kernel, int M, int N, int K, float alpha, float *A,
                float *B, float beta, float *C, cublasHandle_t handle) {
  switch (kernel) {
    case 0: run_cublas(handle, M, N, K, alpha, A, B, beta, C); break;
    case 1: run_naive(M, N, K, alpha, A, B, beta, C); break;
    case 2: run_coalesce(M, N, K, alpha, A, B, beta, C); break;
    case 3: run_smem(M, N, K, alpha, A, B, beta, C); break;
    case 4: run_1d_blocktiling(M, N, K, alpha, A, B, beta, C); break;
    case 5: run_2d_blocktiling(M, N, K, alpha, A, B, beta, C); break;
    case 6: run_vectorized(M, N, K, alpha, A, B, beta, C); break;
    case 9: run_autotune(M, N, K, alpha, A, B, beta, C); break;
    case 10: run_warptiling(M, N, K, alpha, A, B, beta, C); break;
    default:
      printf("未知 kernel 编号: %d\n", kernel);
      exit(EXIT_FAILURE);
  }
}

// ---- 工具函数 --------------------------------------------------------------
void randomize_matrix(float *mat, int n) {
  // 均匀分布在 [-1, 1]:让点积结果数值范围适中,方便和 cuBLAS 逐元素比对
  for (int i = 0; i < n; i++)
    mat[i] = 2.0f * (float)rand() / (float)RAND_MAX - 1.0f;
}

// 逐元素比对:绝对误差或相对误差有一个满足即视为通过(fp32 在 K 很大时会累积误差)
bool verify_matrix(const float *ref, const float *out, int n) {
  for (int i = 0; i < n; i++) {
    double diff = fabs((double)ref[i] - (double)out[i]);
    double rel = diff / (fabs((double)ref[i]) + 1e-6);
    if (diff > 1e-2 && rel > 1e-2) {
      printf("  ✗ 结果不匹配 @ %d: ref=%.5f  out=%.5f  (|diff|=%.5f)\n", i,
             ref[i], out[i], diff);
      return false;
    }
  }
  return true;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("用法: %s <kernel_num> [size] [repeats]\n", argv[0]);
    printf("  kernel_num: 0=cuBLAS  1=naive\n");
    return EXIT_FAILURE;
  }
  int kernel  = atoi(argv[1]);
  int size    = (argc >= 3) ? atoi(argv[2]) : 4096;
  int repeats = (argc >= 4) ? atoi(argv[3]) : 50;

  const int M = size, N = size, K = size;
  const float alpha = 1.0f, beta = 0.0f;

  // ---- 主机端分配 & 初始化 ----
  size_t bytesA = (size_t)M * K * sizeof(float);
  size_t bytesB = (size_t)K * N * sizeof(float);
  size_t bytesC = (size_t)M * N * sizeof(float);
  float *hA   = (float *)malloc(bytesA);
  float *hB   = (float *)malloc(bytesB);
  float *hC   = (float *)malloc(bytesC);
  float *hRef = (float *)malloc(bytesC);
  srand(42);
  randomize_matrix(hA, M * K);
  randomize_matrix(hB, K * N);

  // ---- 设备端分配 & 拷贝 ----
  float *dA, *dB, *dC, *dRef;
  CUDA_CHECK(cudaMalloc(&dA, bytesA));
  CUDA_CHECK(cudaMalloc(&dB, bytesB));
  CUDA_CHECK(cudaMalloc(&dC, bytesC));
  CUDA_CHECK(cudaMalloc(&dRef, bytesC));
  CUDA_CHECK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dC, 0, bytesC));
  CUDA_CHECK(cudaMemset(dRef, 0, bytesC));

  cublasHandle_t handle;
  cublasCreate(&handle);

  // ---- 1) 用 cuBLAS 计算参考结果 ----
  run_cublas(handle, M, N, K, alpha, dA, dB, beta, dRef);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hRef, dRef, bytesC, cudaMemcpyDeviceToHost));

  // ---- 2) 运行被测 kernel 一次并校验 ----
  run_kernel(kernel, M, N, K, alpha, dA, dB, beta, dC, handle);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));

  bool ok = (kernel == 0) ? true : verify_matrix(hRef, hC, M * N);
  printf("Kernel %d  size=%d  正确性: %s\n", kernel, size,
         ok ? "通过 ✓" : "失败 ✗");

  // ---- 3) 性能测试(cudaEvent 计时)----
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  run_kernel(kernel, M, N, K, alpha, dA, dB, beta, dC, handle);  // 预热
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEventRecord(start);
  for (int i = 0; i < repeats; i++)
    run_kernel(kernel, M, N, K, alpha, dA, dB, beta, dC, handle);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  double seconds = ms / 1000.0;
  double flops = 2.0 * (double)M * N * K * repeats;  // 每次 matmul = 2*M*N*K
  double gflops = flops / seconds / 1e9;
  printf("平均耗时: %.3f ms   性能: %.1f GFLOPs/s\n", ms / repeats, gflops);

  // ---- 清理 ----
  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
  free(hA); free(hB); free(hC); free(hRef);
  cudaEventDestroy(start); cudaEventDestroy(stop);
  cublasDestroy(handle);
  return 0;
}
