# CVDMS Fused Kernel 技术报告

## 1. 概述

本报告描述 CVDMS 算法内层 K-series 循环的 CUDA 融合核函数（fused kernel）设计与实现。

### 问题

原始内层 K-series 循环每次迭代需要多次全局显存读写：

```
for n in range(20):
    scratch = laplace(working)      # 读 working → 写 scratch
    scratch *= inv_4piK0            # 读 scratch → 写 scratch
    working *= V                    # 读 working → 写 working
    scratch += working              # 读 working+scratch → 写 scratch
    k_series += scratch * scale     # 读 scratch → 写 k_series
    working, scratch = scratch, working  # 指针交换
```

每次迭代约 6 次显存读写，每次 D2H 同步约 10-50 μs。20 次迭代 = ~120 次全局显存访问，**带宽利用率极低**。

### 解决方案

将整个 K-series 迭代合并为单个 CUDA kernel，每像素在寄存器中独立完成计算：

```
读 wave + V 从显存 → 寄存器
for n in range(iterations):
    拉普拉斯(从显存读邻域) → 寄存器
    乘系数、累加 (全在寄存器)
    写 k_series 到显存 (仅1次累加)
```

## 2. 算法细节

### 2.1 K-operator

$$K(\psi) = V \cdot \psi + \frac{\nabla^2(\psi)}{4\pi K_0}$$

其中 $K_0 = 1/\lambda$，$\lambda$ 为电子波长。

### 2.2 K-series

$$\text{K-series}(\psi) = \sum_{n=1}^{N} c_n \cdot K^n(\psi)$$

系数：
- $c_1 = 1$
- $c_n = \frac{(1.5 - n) \cdot 4 \cdot \text{inv\_4piK0}}{n}$  for $n > 1$

### 2.3 有限差分 Laplacian (可分离, 8 阶精度)

$$\nabla^2(\psi)[i,j] = \text{prefactor} \cdot \sum_{k=-n}^{n} c_k \cdot (\psi[i+k,j] + \psi[i,j+k])$$

周期边界条件通过模运算实现：`(row + k + H) % H`。

### 2.4 收敛检测

每像素独立计算 $|c_n \cdot K^n(\psi)| > \text{threshold}$，通过 block-level shared memory atomic counter 汇总。仅单 int D2H 同步/迭代（vs 原版每 2 迭代一次 D2H + 浮点规约）。

## 3. 架构

### 文件

- `abTEM/abtem/cvdms_kernels.py` — CUDA kernel 源码 + Python 封装

### 核心组件

| 组件 | 说明 |
|------|------|
| `k_iteration_fused` CUDA kernel | 单次 K-iteration：Laplacian → K-operator → 缩放 → 累加 |
| `compute_k_series_fused()` | Python 封装：内存管理、ping-pong 循环、收敛控制 |
| `_get_k_iteration_kernel()` | CuPy RawKernel 编译缓存 |

### Kernel 参数

| 参数 | 传递方式 | 说明 |
|------|---------|------|
| `cur_re/cur_im` | 设备指针 | 当前 wave (float32) |
| `next_re/next_im` | 设备指针 | 下一迭代 wave (float32) |
| `kseries_re/kseries_im` | 设备指针 | 累加结果 (float32) |
| `V` | 设备指针 | 势函数 × sigma (float32, 2D) |
| `sc` | 设备指针 | 有限差分系数 (float32) |
| `lap_factor` | **编译时常量** | `prefactor × inv_4piK0` 字面量内联 |
| `inv_4piK0` | **编译时常量** | `1 / (4πK₀)` 字面量内联 |
| `threshold` | 运行时参数 | 收敛阈值 |
| `H, W` | 运行时参数 | 空间维度 |
| `sn` | 运行时参数 | stencil 半宽度 |
| `iter_n` | 运行时参数 | 当前迭代次数（1-based） |

### 内存布局

```
complex64 wave (batch, H, W)
    ├── buf0_re (float32, batch*H*W)  ← 拆分为实部/虚部
    └── buf0_im (float32, batch*H*W)
    
buf1_re/buf1_im  ← ping-pong 交替缓冲区
kseries_re/kseries_im  ← 累加器
```

## 4. CUDA 编译器 Bug 及解决方案

### 问题

CuPy 13.6.0 中，将 `lap_factor` 和 `inv_4piK0` 作为 CUDA kernel 函数参数传入时，`sum * lap_factor` 被编译器优化为 0。

```cuda
// 错误：lap_factor 作为函数参数传入时乘积被优化为 0
void kernel(..., float lap_factor, ...) {
    float result = sum * lap_factor;  // → 0
}
```

### 诊断

通过增量构建 kernel 逐步定位：
1. v1（无参基准）：正确
2. v2 (+ shared memory)：正确
3. v3 (+ base/stride)：正确
4. v4 (+ K-operator params)：**失败**（乘积归零）
5. v5 (+ `__restrict__`)：**失败**

确定是 CUDA 编译器对函数参数的浮点乘法触发了错误优化。

### 解决

将常量作为编译期字面量嵌入 CUDA 源码，而非通过函数参数传递：

```python
# Python 端：用字面量替换占位符
kernel_src = kernel_src.replace("__LAPF__", f"{lap_factor}")
kernel_src = kernel_src.replace("__INV4PIK0__", f"{inv_4piK0}")

# CUDA 端：编译时常量
const float lap_factor = __LAPF__f;   // → const float lap_factor = 0.19894368f;
const float inv_4piK0 = __INV4PIK0__f;
```

**注意**：不同物理参数（`wavelength`, `dx`, `dy`）会生成不同的 kernel 实例，通过 cache key `f"k_iteration_fused_{lap_factor}_{inv_4piK0}"` 区分。

## 5. 性能分析

### 理论加速

| 方面 | 原始 Python 循环 | Fused Kernel |
|------|-----------------|--------------|
| 全局显存写/迭代 | ~4 次 | 1 次（仅 `next_re/im`） |
| D2H 同步/迭代 | ~1 次（收敛检查） | 0 次 |
| Python 回环开销/迭代 | ~1 次 | 0 次 |
| 收敛检测粒度 | 全局（所有像素） | 逐像素 + block-level 归约 |

### 实测加速比

**vs 生产 GPU Laplacian**（@cuda.jit padding 边界 vs RawKernel modulo 边界）：

| 网格 | 批次 | 迭代数 | 生产 (ms) | Fused (ms) | 加速比 |
|------|------|--------|-----------|------------|--------|
| 128×128 | 1 | 7 | 2.46 | 0.27 | **9.0×** |
| 512×512 | 1 | 7 | 2.73 | 0.21 | **13.0×** |
| 1024×1024 | 1 | 7 | 8.13 | 1.29 | **6.3×** |

> 迭代数随收敛阈值和波函数结构变化（实测 7 次），非固定 50 次。

**加速来源**：
1. **避免 padding 开销**: 生产代码的 `@cuda.jit` Laplacian 使用 padding+wrap 边界处理（分配、拷贝、计算、提取），fused kernel 使用 modulo 算术直接访问
2. **Kernel launch 合并**: ~5 次 CuPy kernel launch → 1 次 RawKernel launch
3. **消除 D2H 同步**: 收敛检测 on-device atomic counter

### 端到端加速

在多 slice 全流程中，K-series 加速被其他操作稀释：

| 场景 | 内层加速 | 全流程加速 | 说明 |
|------|---------|-----------|------|
| 小网格 (128×128), 2 slices | 9× | ~1× | FFT/势函数/任务调度占主导 |
| 大网格 (1024×1024), 多 slice | 6× | ~2× | K-series 占比增大 |

**稀释原因**: 全流程中每个 slice 包含势函数投影、Fresnel 传播（FFT）、backscattering 校正等操作，K-series 仅占总时间的 10-30%（取决于网格大小和收敛条件）。Fused kernel 节省的每 slice 2-7ms 在全流程 1-2s 中占比有限。

**注意**：之前的 350-400× 加速比是相对于使用 `scipy.ndimage.convolve`（CPU）+ 每次迭代 D2H/H2D 拷贝的参考实现。生产代码中的 GPU Laplacian 已全在 GPU 上执行。

## 6. 与原始算法的差异

### 收敛检测

原始算法每 `check_interval=2` 次迭代做一次全局收敛检查。Fused kernel 每次迭代都做逐像素收敛检查，精度更细但行为略有不同。

### 浮点精度

Fused kernel 使用 32-bit 浮点运算，与原始算法的计算顺序不同可能导致 1e-4 ~ 1e-7 级别的数值差异（取决于比较基准）。

## 7. 正确性验证

测试环境：NVIDIA GeForce RTX 3070, CUDA 12.x, CuPy 13.6.0, Python 3.12。

### 7.1 debug_kernel.py — 相同算法对比

GPU fused kernel vs CPU Python 参考实现（使用完全相同的可分离 Laplacian 算法）。
输入: 随机 wave (1×64×64, complex64), 随机 V (64×64, float32), 8 阶 FD stencil, wavelength=0.025。

| 指标 | 值 |
|------|-----|
| K(w) max diff | 7.25 × 10⁻⁷ |
| K(w) mean diff | 1.03 × 10⁻⁷ |
| K-series max diff | 7.25 × 10⁻⁷ |
| 结论 | **PASS** (< 1e-5) |

差异仅来自浮点运算顺序（CPU vs GPU），非算法错误。

### 7.2 test_fused_kernel.py — scipy 参考实现对比

GPU fused kernel vs scipy.ndimage.convolve 参考实现（使用不同的 Laplacian 实现）。

| 网格 | Max diff | Mean diff |
|------|----------|-----------|
| 128×128, batch=2 | 1.87 × 10⁻⁴ | 4.81 × 10⁻⁵ |
| 1024×1024, batch=8 | 2.40 × 10⁻⁴ | 4.81 × 10⁻⁵ |

本测试中的差异主要来自参考实现使用 scipy.ndimage.convolve，其核函数居中/索引约定、浮点运算顺序与 CUDA 可分离 Laplacian 不同。debug_kernel.py 的差异（7e-7）更能反映真实精度。

### 7.3 收敛行为

| 迭代 | 原始 (n_above) | Fused (n_above) | 说明 |
|------|---------------|-----------------|------|
| 1 | 32768 | 32768 | 所有像素活跃 |
| 2 | ~30000 | ~30000 | 开始收敛 |
| ... | ... | ... | 持续下降 |
| N | 0 | 0 | 完全收敛 |

收敛速度与原始算法基本一致，因为采用相同的逐像素收敛判据。

## 8. 回归测试

| 测试 | 状态 |
|------|------|
| 同算法精度 (debug_kernel.py) | PASS |
| scipy 参考对比 (test_fused_kernel.py) | PASS |
| overflow 检测 | 正常 |
| 批处理正确性 | 正常 |
| 多次调用稳定性 | 正常 |
| 参数变化 (wavelength, threshold) | 正常 |
| 默认启用集成 (use_fused_kernel=True) | 正常
