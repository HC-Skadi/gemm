# CUDA SGEMM 优化学习

跟着 siboehm 的经典文章 [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM) 一步步把一个矩阵乘 kernel 从「naive」优化到接近 cuBLAS。

计算目标(SGEMM):`C = alpha * (A @ B) + beta * C`,其中 A 是 M×K,B 是 K×N,C 是 M×N,全部行主序。

## 优化路线图(文章为 RTX A6000 数据;「A100 实测」是我的云 GPU 数据)

| # | Kernel | 文章 GFLOPs/s | A100 实测 | 占我的 cuBLAS | 状态 |
|---|--------|----------|-----------|-----------|------|
| 0 | cuBLAS 基线 | 23249 | 16575.2 | 100% | ✅ |
| 1 | Naive | 309 | 291.8 | 1.8% | ✅ |
| 2 | GMEM 合并访问 | 1986 | 2528.3 | 15.3% | ✅ |
| 3 | SMEM 缓存分块 | 2980 | 5279.7 | 31.9% | ✅ |
| 4 | 1D Blocktiling | 8474 | 9215.9 | 55.6% | ✅ |
| 5 | 2D Blocktiling | 15971 | 待测 | ? | ✅ 本课 |
| 6 | 向量化访存 | 18237 | 待测 | ? | ✅ 本课 |
| 9 | Autotuning | 19721 | | | ⬜ |
| 10 | Warptiling | 21779 | | | ⬜ |

> 两块卡互补:A100 带宽大、L2 大(40MB),memory-bound 的 kernel 反超文章;
> A6000 FP32 峰值高(~2×),算力-bound 的 kernel 和 cuBLAS 绝对值更高。
> 所以重点看**占自己 cuBLAS 的百分比**和**每步的提速倍数**。

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

### Lesson 2 — GMEM 合并访问

**代码**:[kernels/02_coalesce.cuh](kernels/02_coalesce.cuh)

**唯一的改动**:线程→元素映射,让「同一 warp 内相邻线程」对应「C 的相邻列」。数学、工作量、线程数都没变,却快了约 6 倍。

**核心心智模型**:合并看的是「固定第 i 次迭代时,一个 warp 的 32 个线程同时访问的地址是否连续」,而不是单个线程自己循环时是否连续。

| 访问(固定 i,看整个 warp) | naive | 合并版 |
|------|------|------|
| A | 跨行,stride K ❌ | cRow 相同 → 广播 ✓ |
| B | col 相同 → 广播 | cCol 连续 → 合并 ✓ |
| C | 跨行,stride N ❌ | cCol 连续 → 合并 ✓ |

**为什么还是只有 ~8.5%**:虽然访存合并了,但每个 A/B 元素仍被反复从**慢速全局内存**重复读取(A 的每一行被 N 个线程各读一遍),算术强度依旧极低,仍是 memory-bound。→ Kernel 3 用**共享内存**把 tile 缓存到片上,消除重复的全局内存读取。

**对比命令**:重跑上一课的 sector 指标,应从接近 32 掉到接近 4:
```bash
ncu --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio ./sgemm 2 4096 1
```

### Lesson 3 — SMEM 缓存分块

**代码**:[kernels/03_smem.cuh](kernels/03_smem.cuh)

**痛点**:kernel 2 里每个 A/B 元素被反复从慢速全局内存重复读取。

**思路**:一个 block 里的线程共用同一片 A/B 数据,那就沿 K 维分块,把每段 tile 先协作搬进片上共享内存(SMEM,约快 100×),大家从 SMEM 反复读。

**流程**(每段 K):协作加载(合并)→ `__syncthreads()` → 从 SMEM 做部分点积 → `__syncthreads()` → 推进。两个 barrier 缺一不可:第一个保证数据齐了再算,第二个保证算完了再覆盖 SMEM。

**收益**:每个 SMEM 元素被 BS=32 个线程复用 → 全局内存流量降为 1/32,算术强度提升。

**为什么还是只有 ~13%**:瓶颈从"全局带宽"转移到"**SMEM 带宽 / MIO 队列**"。内层每做 1 次 FMA 要发 2 次 SMEM load,访存指令把计算单元堵住了。→ Kernel 4 让每个线程算多个输出,把从 SMEM 读到寄存器的值复用到多次 FMA,摊薄 SMEM 访问。

**A100 提示**:A100 有 40MB L2(A6000 只有 6MB),kernel 2 的重复读大量命中 L2 已经很快,所以 kernel 3 相对提升会比文章的 1.5× 更温和(预计 ~3000–3800 GFLOPs)。

**验证**:全局流量应下降(dram 吞吐比 kernel 2 低),瓶颈变成 SMEM:
```bash
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./sgemm 3 4096 1
ncu --set full -o report_k3 ./sgemm 3 4096 1   # 看 Warp State,会出现 MIO Throttle 停顿
```

### Lesson 4 — 1D Blocktiling

**代码**:[kernels/04_1d_blocktiling.cuh](kernels/04_1d_blocktiling.cuh)

**痛点**:kernel 3 内层 1 FMA : 2 SMEM load,卡在 MIO Throttle。

**手法**:每个线程算 TM=8 个输出(竖条)。内层把 dotIdx 放最外层,先把 `Bs` 的值读进寄存器 `tmpB`,再用它连做 TM 次 FMA。SMEM load : FMA 从 2:1 降到 (TM+1):TM = 9:8;8 个结果累加在寄存器数组里。

**分块**:BM=BN=64, BK=8, TM=8 → 线程数 = 64×64/8 = 512。

**两套索引**:`innerRow/Col*`(加载,为 GMEM 合并)vs `threadRow/threadCol`(计算,铺在 64×64 输出上)。同一线程在搬数据和算数据时角色不同,别混。

**A100 实测**:9215.9 GFLOPs(14.913 ms)= k3 的 1.75×,达到自己 cuBLAS 的 **55.6%**(文章同期仅 36.5%,我们进度领先),约 FP32 峰值 19.5T 的 47.3%。
相对增幅小于文章的 2.85× 是因为 A100 把 k3 抬得偏高(带宽/L2 优势),看百分比才公平。

### Lesson 5 — 2D Blocktiling

**代码**:[kernels/05_2d_blocktiling.cuh](kernels/05_2d_blocktiling.cuh)

**痛点**:k4 每线程一竖条 TM=8,SMEM load : FMA = 9:8,A 值用一次就扔。

**手法**:每线程负责 TM×TN = 8×8 = 64 个输出的小方块。每个 dotIdx 先搬 TM+TN = 16 个值进寄存器 tmpA/tmpB,再做 64 次**外积** FMA → SMEM load : FMA = 16:64 = **1:4**。tmpA 被复用 TN 次、tmpB 被复用 TM 次,双重复用。

**分块**:BM=BN=128, BK=8, TM=TN=8 → 线程数 = 128×128/64 = 256;每线程搬 4 个 A + 4 个 B(tile 线性编号,「恰好 1 个」的约束不再成立)。

**代价**:~80+ 寄存器/线程(64 累加器 + tmpA/tmpB),occupancy 会被寄存器卡住 —— Lesson 4 问答里讲的 blockDim/occupancy 讲究在这里真实上演,可用 `ncu --section Occupancy` 观察寄存器是否成为 limiting factor。

**A100 实测**:待填。文章 A6000 是 k4 的 1.88×(8474→15971)。

---

### 阶段小结 — Kernel 3/4/5

三个 kernel 是同一条主线:把数据沿存储层级上移,每上移一层,复用次数提高一个量级。

```
寄存器   ~256KB/SM    ~0 周期     k4 的 tmpB、k5 的 64 个累加器
SMEM     164KB/SM    ~30 周期    k3 起 tile 的存放处
L2       40MB        ~200 周期   k2 已受益(A100 40MB vs A6000 6MB)
HBM      80GB ~2TB/s ~400+ 周期  所有优化的起点
```

| | k3 SMEM 分块 | k4 1D Blocktiling | k5 2D Blocktiling |
|---|---|---|---|
| 解决的瓶颈 | GMEM 重复读 | SMEM 带宽 / MIO(1 FMA : 2 load) | SMEM 流量(9 load : 8 FMA) |
| 手法 | tile 搬进 SMEM,block 内共用 | 每线程 TM=8 个输出,B 值进寄存器复用 8 次 | 每线程 8×8=64 个输出,tmpA、tmpB 做外积 |
| 内层 SMEM load : FMA | 2 : 1 | 9 : 8 | 16 : 64 = 1 : 4 |
| 算术强度(FLOP/Byte)¹ | 8 | 16 | 32 |
| A100 实测(占 cuBLAS) | 5279.7(31.9%) | 9215.9(55.6%) | 待测 |

¹ 每 BK 步搬 (BM+BN)·BK 个元素、做 BM·BN·BK 次 FMA。每 FMA 的 GMEM 流量 = 4B·(1/BM + 1/BN),对应 2 FLOP。tile 边长 32→64→128,强度 8→16→32。

1. k3 之后 K 循环的骨架(加载 → sync → 算 → sync)没有再变,变化都在内层寄存器的用法。GMEM 端只有 tile 尺寸在变:32→64→128,流量逐次减半。
2. A100 屋顶线拐点 ≈ 19.5 TFLOPS ÷ 2 TB/s = 10 FLOP/Byte。k3(8)在拐点左侧,带宽受限;k4(16)刚过拐点;k5(32)已在算力一侧。过了拐点不等于接近峰值:SMEM 指令数(MIO)、寄存器压力、bank conflict 仍在起作用,分别对应后面的 k6 向量化、k9 扫参数、k10 warptiling。
3. 瓶颈逐级转移:GMEM 重复读 →(k3)SMEM 指令太多 →(k4)A 值只复用一次 →(k5)寄存器压力与指令数。

### Lesson 6 — 向量化访存(float4)

**代码**:[kernels/06_vectorized.cuh](kernels/06_vectorized.cuh)

**痛点**:k5 加载阶段每线程每 BK 步发 8 条 32 位 load(4 个 A + 4 个 B),指令多,32 位传输也用不满内存通路。

**手法**:GMEM→SMEM 改用 float4(128 位)。BM·BK/4 = 256 个 float4、BK·BN/4 = 256 个,与 256 线程一一对应:每线程 1 条 LDG.128 搬 A、1 条搬 B,加载指令 8 条 → 2 条。SMEM 端同样 128 位写入。附带好处:向量的行列坐标在 K 循环中不变,加载用的 div/mod 从循环里移出,只算一次。

**对齐前提**:K、N 为 4 的倍数(GMEM 行首 16B 对齐);SMEM 数组声明 `__align__(16)`,行跨度 32B/512B 均为 16 的倍数。约束:BM·BK/4 == BK·BN/4 == 线程数,改参数时要保持。

**验证**:SASS 里应出现 128 位 load:
```bash
cuobjdump -sass sgemm | grep -E 'LDG\.E' | sort | uniq -c
# k6 的 kernel 里应是 LDG.E.128,不再是 32 位
```

**A100 实测**:待填。文章 A6000 是 k5 的 1.14×(15971→18237);A100 带宽充裕、加载占比更小,增幅可能低于文章。
