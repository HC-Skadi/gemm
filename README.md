# CUDA SGEMM 优化学习

跟着 siboehm 的经典文章 [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM) 一步步把一个矩阵乘 kernel 从「naive」优化到接近 cuBLAS。

计算目标(SGEMM):`C = alpha * (A @ B) + beta * C`,其中 A 是 M×K,B 是 K×N,C 是 M×N,全部行主序。

## 优化路线图(参考文章在 RTX A6000 上的数据)

| # | Kernel | GFLOPs/s | 占 cuBLAS | 状态 |
|---|--------|----------|-----------|------|
| 0 | cuBLAS 基线 | 23249 | 100% | ✅ |
| 1 | Naive | 309 | 1.3% | ✅ 本课 |
| 2 | GMEM 合并访问 | 1986 | 8.5% | ⬜ |
| 3 | SMEM 缓存分块 | 2980 | 12.8% | ⬜ |
| 4 | 1D Blocktiling | 8474 | 36.5% | ⬜ |
| 5 | 2D Blocktiling | 15971 | 68.7% | ⬜ |
| 6 | 向量化访存 | 18237 | 78.4% | ⬜ |
| 9 | Autotuning | 19721 | 84.8% | ⬜ |
| 10 | Warptiling | 21779 | 93.7% | ⬜ |

> 你的云 GPU 若不是 A6000,绝对数字会不同,重点看**每一步相对上一步的提速倍数**和 profiler 指标。

## 编译 & 运行

```bash
# 1) 按你的 GPU 架构编译(见 Makefile 顶部注释:A100=80, T4=75, A6000=86...)
make CUDA_ARCH=80

# 2) 运行:./sgemm <kernel> [size] [repeats]
./sgemm 0            # cuBLAS 基线,4096x4096
./sgemm 1 2048 10    # naive 很慢,先用小一点的尺寸和更少的重复次数
```

输出会打印:正确性(与 cuBLAS 逐元素比对)+ 平均耗时 + GFLOPs/s。

## Profiling(Nsight Compute)

理解每个 kernel「为什么慢」的关键工具。几个最有用的命令:

```bash
# 看显存吞吐(naive 会非常低,合并访问后会飙升)
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./sgemm 1 4096 1

# 看访存是否合并:每次全局 load 平均用了多少个 32B sector(越低越好,理想 4)
ncu --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio ./sgemm 1 4096 1

# 完整报告(信息量大,建议导出后在 GUI 里看)
ncu --set full -o report_k1 ./sgemm 1 4096 1
```

---

## 课程笔记

### Lesson 1 — Naive

**代码**:[kernels/01_naive.cuh](kernels/01_naive.cuh)

**并行策略**:每个线程算 C 的一个元素,读 A 的一行 × B 的一列做 K 次乘加。线程互相独立,无需同步。

**为什么慢(~1.3% cuBLAS)**:
1. **访存不合并(uncoalesced)**:代码里 `threadIdx.x` 映射到「行 row」。同一个 warp 的 32 个线程 row 连续、col 相同,它们读的 `A[row*K+i]` 和写的 `C[row*N+col]` 地址相差一整行(K/N 个 float),硬件无法把它们合并成一次宽事务 → 显存带宽被浪费(文章实测仅 ~15 GB/s)。
2. **算术强度极低**:每做 1 次乘加(2 FLOP)就要从全局内存读 2 个 float(8 字节)。计算被访存彻底卡住(memory-bound)。

**下一课预告**:Kernel 2 只是把 row/col 的线程映射对调一下,让同一 warp 读连续地址,吞吐就能从 ~15 GB/s 跳到 ~110 GB/s,性能提升约 6 倍。
