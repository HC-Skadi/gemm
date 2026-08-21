# GEMM 学习项目 Makefile
#
# 参考文章: https://siboehm.com/articles/22/CUDA-MMM
#
# 用法:
#   make                       # 编译(默认 sm_80,即 A100)
#   make CUDA_ARCH=86          # 指定 GPU 架构后编译
#   ./sgemm <kernel> [size] [repeats]
#
# 常见云 GPU 的 CUDA_ARCH:
#   T4      = 75      V100 = 70      A100 = 80
#   A6000/RTX30xx = 86           L4/RTX40xx = 89           H100 = 90

NVCC      := nvcc
CUDA_ARCH ?= 80

NVCC_FLAGS := -O3 -std=c++17 -arch=sm_$(CUDA_ARCH)
LDFLAGS    := -lcublas

TARGET := sgemm
SRC    := sgemm.cu

$(TARGET): $(SRC) $(wildcard kernels/*.cuh)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(TARGET)
