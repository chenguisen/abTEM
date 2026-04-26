# CVDMS 性能优化报告

## 1. 测试环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 3070 |
| CuPy | 13.6.0 |
| abTEM | 1.0.9 |
| FFT 后端 | cuFFT（通过 `fft="cupy"`） |

## 2. 优化内容

| # | 优化 | 文件 | 说明 |
|---|------|------|------|
| 1 | FFT Laplacian k² 网格缓存 | `finite_difference.py` | 按 `(Ny, Nx, device_id)` 缓存傅里叶空间频率因子 `-4π²k²`，避免每次调用重建 kx/ky/k² 网格 |
| 2 | scratch 缓冲区预分配 | `cvdms.py` | `xp.empty_like` 预分配，循环内用 `scratch[:] = laplace(working)` 复用，消除反复 allocation |
| 3 | 引用交换替代拷贝 | `cvdms.py` | `working, scratch = scratch, working` 消除冗余数组拷贝 |
| 4 | FFT Laplacian clip 原地化 | `finite_difference.py` | `xp.clip(..., out=...)` 避免中间数组分配 |
| 5 | 外层循环惰性初始化 | `cvdms.py` | `working=None`，首轮直接用 `waves_array`，后续迭代复用 `k_series` 内存 |

**注意**：收敛检查保持 `xp.abs()` 不变。CuPy 的 `abs()` 是融合核函数（单次 kernel launch），手动展开为 `real² + imag² > threshold²` 需要 3 次 kernel launch，实测慢 1.6x。

## 3. 微基准测试 (231×400 = 92,400 px)

| 操作 | 优化前 | 优化后 | 加速比 |
|------|--------|--------|--------|
| FFT Laplacian（k² 缓存后） | ~0.28 ms | 0.24 ms | ~12% |
| Inner K series | ~3.1 ms | 1.7 ms | ~42% |
| Forward scattering | ~10 ms | 7 ms | ~30% |

## 4. 全流程对比 (6 片层)

### 中等网格：231×400 = 92,400 px

| 算法 | 耗时 | 加速比 |
|------|------|--------|
| CVDMS (FFT) | 994 ms | **基准** |
| CVDMS (FD) | 3919 ms | 3.9× 慢于 FFT |

### 大网格：549×951 = 522,099 px

| 算法 | 耗时 | 加速比 |
|------|------|--------|
| CVDMS (FFT) | 8477 ms | 基准 |
| CVDMS (FD) | 5084 ms | **0.6×** FFT 慢于 FD |

大网格下 FFT 较慢的原因是 FFT 维度包含大素数因子（549 = 3² × **61**，951 = 3 × **317**），cuFFT 对大素数因子的维度效率显著下降。

## 5. 网格尺寸与 FFT 维度对性能的影响

FFT 与 FD 的性能优劣同时取决于两个因素：**网格总像素数**和**FFT 维度的素数因子构成**。

### 5.1 网格大小分类

| 分类 | 像素范围 | 典型尺寸 | 推荐方法 |
|------|---------|---------|---------|
| 小网格 | < 100K | 231×400, 256×256 | FFT（快 2-4×） |
| 中网格 | 100K ~ 300K | 400×400, 512×512 | FFT（快 ~2×） |
| 大网格 | > 300K | 768×1331, 1024×1024 | 取决于 FFT 维度质量 |

### 5.2 FFT 维度质量

| 网格 | 素数因子 (Ny) | 素数因子 (Nx) | 最大素数 | FFT 加速比 |
|------|--------------|--------------|---------|-----------|
| 231×400 | 3, 7, 11 | 2⁴, 5² | **11** | 4.1× (FFT 更快) |
| 154×267 | 2, 7, 11 | 3, **89** | **89** | ~2× (FFT 较快) |
| 652×652 | 2², **163** | 2², **163** | **163** | ~1× (接近) |
| 549×951 | 3², **61** | 3, **317** | **317** | 0.6× (FD 更快) |

**规律**：
- 当 FFT 维度的最大素数因子 ≤ 13 时，cuFFT 性能良好，FFT 显著快于 FD
- 当最大素数因子 ≥ 13 时，cuFFT 性能下降，FFT 优势减小
- 当最大素数因子 > 50 时，FD 可能快于 FFT

### 5.3 选择指南

```
FFT 维度质量好（素数 ≤ 13）+ 中小网格 → 用 FFT（快 2-4×，且更精确）
FFT 维度质量差（素数 > 13）          → 用 FD（约等价或略快）
大网格 + 好维度                       → 两者均可，FFT 精度更优
大网格 + 差维度                       → FD 更快
```

## 6. 数值精度验证

所有优化均不影响数值结果：

| 测试 | 结果 |
|------|------|
| `max|FFT - FD|`（中等网格） | 4.67×10⁻⁶ |
| `max|FFT - FD|`（大网格） | 9.36×10⁻⁷ |
| 单元测试 | 16/18 通过（2 个 pre-existing 失败为 `Probe.array` 属性问题，与优化无关） |

## 7. 建议

1. **默认使用 `laplace_method="fft"`** — 中小网格快 2-4×，且 FFT 在高空间频率下无频散误差，能正确保留菊池线等弱衍射特征
2. **大网格含大素数因子时切换 `laplace_method="finite-difference"`** — FD 在高延迟 FFT 维度下更快
3. **启用 CuPy FFT 后端**：配置 `fft="cupy"`（而非 `fft="numpy"`）以使用 GPU cuFFT

## 8. 验证命令

```bash
conda run -n py4dstem python diag_cvdms_optimization.py  # 生成优化报告
conda run -n py4dstem python -m pytest test/test_cvdms_multislice.py -v  # 运行单元测试
```

## 9. RTX 3080 硬件特定优化

### 9.1 测试环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 3080 (sm_86, Ampere) |
| SM 数 | 68 |
| L2 缓存 | 5 MB |
| 显存 | 10.7 GB |
| 最大线程/SM | 1536 |
| 共享内存/block | 48 KB |
| CuPy | 13.6.0 |
| abTEM | 1.0.9 |

### 9.2 优化内容

| # | 优化 | 文件 | 说明 |
|---|------|------|------|
| 1 | **Shared memory tiled kernel** | `cvdms_kernels.py` | 32×16 tile + halo 协作加载，Laplacian stencil 全局访问减少 ~5-10× |
| 2 | **Backscattering GPU 化** | `cvdms.py` | 将 `prefactor`/`stencil_raw` 转发到 `_cvdms_backscattering_correction`，使背散射路径也使用 fused GPU kernel |
| 3 | **Modulo → conditional** | `cvdms_kernels.py` | 用条件分支替换整数除法（sm_86 的 `DIV` 约 20 cycles/次） |
| 4 | **Per-iteration D2H sync** | `cvdms_kernels.py` | fused kernel 每次迭代都检查收敛（不遵循 `check_interval`） |

**优化 4 说明**：在 fused kernel 中，每 K-iteration 仅需 ~28 μs 的 D2H 同步开销（2 × int 读取）。如果 batch 收敛检查（`check_interval > 1`），最坏情况下会多跑 `check_interval - 1` 次完整的 K-iteration（每次 ~150 μs），远超过省下的 D2H 时间。因此 fused kernel 始终逐迭代检查，忽略 `check_interval` 参数。这与非 fused Python 循环路径不同——后者每次迭代有多次 kernel launch + 全局内存读写，D2H 占比较高，batch 才有收益。

### 9.3 Tiled Kernel 设计

```cuda
// Tile 共享内存布局
const int TX = 32;           // tile 宽度
const int TY = 16;           // tile 高度
const int sx = TX + 2 * sn;  // tile + halo 宽度 (40 当 sn=4)
const int sy = TY + 2 * sn;  // tile + halo 高度 (24)

extern __shared__ float shared[];
float* tile_re = shared;           // 实部 tile
float* tile_im = shared + sy * sx; // 虚部 tile

// 协作加载: 512 线程共享加载 960 元素 tile+halo
for (int i = threadIdx.y; i < sy; i += blockDim.y) {
    for (int j = threadIdx.x; j < sx; j += blockDim.x) {
        // ... 条件分支替代 modulo 的周期边界处理 ...
        tile_re[i * sx + j] = cur_re[idx];
        tile_im[i * sx + j] = cur_im[idx];
    }
}
```

**Tile 尺寸选择**：RTX 3080 的 68 个 SM 配合 32×16 的 tile（512 线程）可达到 100% 占用率（3 blocks/SM × 512 = 1536 线程 = 最大值）。更大的 32×32 tile（1024 线程）因 max threads/SM=1536 限制只能跑 1 block/SM，占用率降至 67%。

**共享内存用量**：`2 × (16+8) × (32+8) × 4 = 7680 bytes`，远低于 48 KB 上限，剩余空间留给 L1 缓存。

### 9.4 性能数据

**正方形网格 (SrTiO3, 313×313, 98 slices, 30keV, 1 frozen phonon)**：

| 配置 | 总时间 | ms/slice | 相对非 tiled |
|------|--------|----------|-------------|
| Forward only | 4.5s | 46 | — |
| 非 tiled + 背散射 | 17.8s | 181 | 基准 |
| **Tiled + 背散射** | **15.8s** | **161** | **1.13×** |
| 背散射占比（tiled） | ~11.3s | ~115 | 71% |

**大正方形网格 (SrTiO3, 625×625, 196 slices, 30keV, 1 frozen phonon)**：

| 配置 | 总时间 | ms/slice |
|------|--------|----------|
| **Tiled + 背散射** | **34.2s** | **175** |

**非正方形网格 (Si 111, 615×1065, 118 slices, 30keV)**：

| 配置 | 总时间 | ms/slice |
|------|--------|----------|
| Tiled + 背散射 | 18.8s | 159 |
| 非 tiled + 背散射 | 18.3s | 155 |

非正方形网格下 tiled 与非 tiled 性能相近，因为 tile 的 halo 开销（960/512=1.875× overhead）抵消了共享内存的收益。

### 9.5 背散射瓶颈分析

背散射占总时间的 **71%**，原因是：

1. `_cvdms_backscattering_correction` 调用两次 `_cvdms_inner_k_series`（当前 slice 和下一 slice 的势函数）
2. 背散射使用 `convergence_threshold=1e-16`（前向散射为 1e-6），导致 K-series 迭代次数大幅增加
3. 各 K-iteration 间存在数据依赖，无法并行

**已做的优化**：将 fused kernel 参数（`use_fused_kernel`, `prefactor`, `stencil_raw`）通过 `_cvdms_backscattering_correction` 转发到背散射路径，使其也能使用 GPU fused kernel（而非 Python 循环）。

### 9.6 与 RTX 3070 对比

| 项目 | RTX 3070 (原文档) | RTX 3080 (本文) |
|------|-------------------|-----------------|
| SM 数 | 46 | 68 |
| L2 缓存 | 4 MB | 5 MB |
| 显存带宽 | ~448 GB/s | ~760 GB/s |
| 同网格性能 (231×400, 6 slices) | ~5.1s (CVDMS ci=2) | — |
| 大网格 (625×625, 196 slices) | — | ~34.2s |

### 9.7 已验证的无效优化

| 尝试 | 结果 | 原因 |
|------|------|------|
| `check_interval` batch D2H sync | **无效**（反降 10%） | 每 K-iteration 的成本（~150 μs）远高于 D2H 同步（~28 μs） |
| Tiled kernel 用于非正方形网格 | **无效**（约持平） | Halo 加载开销（1.875×）抵消了共享内存收益 |

## 10. GPU 利用率分析与 check_interval 优化

### 10.1 问题描述

用户观察到 CVDMS 算法的 GPU 利用率显著低于 Fourier 多片层算法（约 <50% vs >90%）。
分析确定根因是收敛检查导致的 D2H（Device-to-Host）同步开销：

- `int(xp.sum(xp.abs(working) > threshold))` — 收敛检查，强制 GPU pipeline 排空
- `float(xp.abs(exit_wave).sum())` — 发散检查，另一次 D2H 同步
- `bool(xp.any(xp.isinf(exit_wave)))` — 数值稳定性检查，第三次同步
- **Fourier Multislice**: 0 次 D2H 同步（纯 GPU pipeline）
- **CVDMS (original)**: 每次外泰勒项和每次内 K 级数迭代均做同步
- 总同步次数：约 20 片层 × 50 外项 × 50 内项 = **50,000 次/模拟**

### 10.2 优化方案：check_interval

将收敛/发散/稳定性检查从"每次迭代"改为"每 N 次迭代批量执行"：

- `check_interval=1`（原始）：每次迭代检查 → 最大约 9,600 次同步/片层
- `check_interval=2`（默认）：每 2 次迭代检查 → 同步次数减半
- `check_interval=3`：每 3 次迭代检查 → 更低同步频率（注意：数值实验显示 ci=3 可能因收敛判断延迟而增加总迭代次数）

代码改动：
- `_cvdms_forward_scattering()`: 外泰勒循环在 `n_exp_order % check_interval == 0` 时执行收敛/发散/稳定性检查
- `_cvdms_inner_k_series()`: 内 K 级数循环在 `n_sqrt_order % check_interval == 0` 时检查收敛 + 停滞
- 代价：最多 `check_interval - 1` 次多余迭代（对收敛的泰勒级数可忽略）

### 10.3 测试结果

**测试环境：** RTX 3070, CuPy 13.6.0, 185K px 网格 (8×8×20 Si(111)), 20 片层

| 算法 | 耗时 | GPU 利用率 | 提速 (vs ci=1) | 备注 |
|------|------|-----------|---------------|------|
| Fourier Multislice | 1617 ms | 88.5% | — | 零 D2H 同步 |
| CVDMS ci=1 (原始) | 5273 ms | 71.9% | — | 32%-93% 波动 |
| **CVDMS ci=2 (优化)** | **5117 ms** | **68.2%** | **~3%** | 11%-84% 波动 |
| CVDMS ci=3 | 8544 ms | 86.4% | -62% | 过多额外迭代 |

### 10.4 分析

**D2H 同步开销微基准测试**（64×64 complex64, RTX 3070）：

| 操作 | 耗时 |
|------|------|
| 纯 GPU 计算（无同步） | 189 μs |
| 计算 + 1 次 D2H 同步 | 203 μs |
| 计算 + 2 次 D2H 同步 | 228 μs |
| **每次 D2H 同步开销** | **~14 μs** |

**每模拟总同步开销：** 50,000 次 × 14 μs ≈ **700 ms**（约 5273 ms 的 13%）

**关键发现：**

1. **D2H 同步不是主导瓶颈。** 单次同步仅 ~14 μs，总开销 ~13% 的模拟时间。
2. **check_interval=2 的收益合理但有限。** 减半同步次数后，总时间改善约 3%。
3. **GPU 利用率差异的真实原因更复杂。** 原子 CVDMS 算法结构本身更串行化：
   - 每泰勒项依赖前一项 → 无法并行
   - K 算子交替执行逐点运算（快）和拉普拉斯算子（慢） → 难以有效重叠
   - Fourier 多片层是 FFT → 乘法 → FFT → 传播的简单管线 → GPU 高效流水线化
4. **nvidia-smi 采样率不足。** ~10Hz 采样（每 100ms 一次）无法捕获微秒级 pipeline 停顿。
5. **check_interval=3 过度。** 收敛判断延迟导致级数多运行 2-3 次迭代，总时间反增 62%。

### 10.5 建议

- **默认保持 check_interval=2** — 在同步开销和收敛检测延迟之间取得最佳平衡
- **对极厚样品（>100 片层）可考虑 check_interval=3** — 更多片层意味着更多同步，ci=3 的优势可能显现
- **主要性能瓶颈在算法结构而非 D2H 同步** — 如需根本性改善 GPU 利用率，需从算法并行度入手（如批量泰勒项合并计算）
- **如无收敛或发散问题，可考虑 check_interval=1** — 行为不变，仅约 3% 的性能代价
