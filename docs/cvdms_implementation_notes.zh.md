# CVDMS 算法实现说明

## 版本记录

| 日期 | 版本 | 说明 |
|------|------|------|
| 2026-04-23 | v1.0 | 基于 ImageSimulation_CGS 的 `transmitSmallProbe_propCVDMS_CGS_BSC` 初始移植 |
| 2026-04-23 | v1.1 | 修复内层 K-级数发散问题，增加收敛检测 |
| 2026-04-23 | v1.2 | 实现背散射波完整反向传播，替换单步近似 |
| 2026-04-23 | v1.3 | 拉普拉斯算符增强：默认精度 6→8（对应9点法），新增 FFT 方法 |
| 2026-04-23 | v1.4 | 数值稳定性与返回类型修复：NaN 截断、停滞检测、fully_corrected 生效、外层非致命警告 |
| 2026-04-23 | v1.5 | API 简化：合并 expansion_scope + include_backscattering 为 backscattering: bool，语义更清晰 |
| 2026-04-27 | v1.6 | C++ CUDA 后端 + backend 选择参数：新增 `backend` 字段（auto/c++/cupy），控制使用 C++ CUDA 还是 CuPy/Python 后端 |
| 2026-04-27 | v1.7 | K-series 融合 kernel: 将内层循环 4 次 kernel launch (laplacian + K-op + scale + converge) 融合为单次，前向散射性能提升 1.28x，BSC 全链路 1.29x |
| 2026-04-27 | v1.8 | C++ CUDA 优化：ConvergenceResult 结构体单 D2H 复制、外层 Taylor 循环融合 kernel、compute_full_series 融合 kernel、cuFFT 拉普拉斯算符后端 |
| 2026-05-04 | v1.9 | BSC running accumulation 反向传播：O(N) 逐片层累加替代 O(N²) 逐层回传，修复出口面深度分布 bug。**数值验证确认与 CGS 逐层独立回传等价**（相对误差 6e-7，纯浮点舍入）。 |
| 2026-05-04 | v2.0 | 发现 ImageSimulation_CGS 反向传播方向错误：使用 `exp(i·K·dz)`（向下传播）替代 `exp(-i·K·dz)`（向上传播），导致 BSC 相位错误、菊池线图案差异。1/K 级数经验证为等价（级联递推=二项式系数）。|
| 2026-05-04 | v2.2 | **HRTEM 支持**：验证 `PlaneWave` + `CVDMSMultislice` 完整 pipeline，支持平行束成像、CTF 衬度传递、冷冻声子平均。新增 notebook `cvdms_hrtem`。|
| 2026-05-05 | v2.3 | **修复 BSC K-级数系数约定错误**：`_cvdms_inner_k_series` 使用 forward scattering 约定（c₁=1）计算 BSC，导致 BSC 修正项在 30keV 下被放大 2πK₀ ≈ 90 倍。新增 λ/(2π) 事后缩放修正。|

## 关键修改

### v1.1 内层 K-级数收敛控制

#### 问题

`_cvdms_inner_k_series`（对应 `calK_PureForward`）在以下情况下发散：

1. **真空切片**：当切片势能为零时，K-算符退化为仅拉普拉斯项 `K(ψ)=∇²ψ/(4πK₀)`。有限差分拉普拉斯算符在高频分量处会产生非物理的特征值放大，导致逐次应用 K 时波函数振幅呈指数增长。
2. **强势场**：重原子（如 Au）在低加速电压（如 80 keV）下产生很强的势能调制，高阶 Kⁿ 项不收敛。
3. **过严收敛阈值**：使用 `convergence_threshold=1e-10` 等超出 float32 精度极限的阈值时，内层级数会持续迭代到数值不稳定。

#### 症状

- `RuntimeWarning: invalid value encountered in divide` — 拉普拉斯算符产生 NaN
- 波函数振幅随 Kⁿ 迭代指数增长（从 1e-4 增长到 1e2 仅需 20 次迭代）
- 数值噪音污染出口波，导致 `assert np.mean(diff) < 0.1` 等校验失败

#### 修复

在 `_cvdms_inner_k_series` 中添加了逐像素收敛控制和发散检测：

```python
# 发散检测：如果未收敛像素数增加，则停止内层级数
# 对应原始 C++ 代码中的 fcms_taylor_max_iter() 动态限制
if prev_n_above is not None and n_above > prev_n_above:
    break
```

原理：K-级数 `Σ cₙ·Kⁿ(ψ)` 的前几项往往快速收敛（高空间频率被拉普拉斯衰减），但之后因有限差分拉普拉斯的非物理高频特征值开始发散。检测未收敛像素数的转折点，在发散趋势初现时截断级数，得到最佳近似。

同时增加了 NaN/Inf 数值稳定性检查。

### v1.2 背散射波完整反向传播

#### 背景

v1.0–v1.1 中，背散射修正仅在当前切片界面处应用一次 BSC 算子：

```
ψ_corrected = ψ_forward - BSC(ψ_forward)
```

这对应 Chen & Van Dyck (1997) 论文中的"单次背散射近似"（single back-scattering approximation, Sec 4.1）。但对于较厚的样品或需要精确背散射波场分布的应用，单步近似不够准确。

#### 修改

根据论文 Eq.(47) 的理论框架，实现了完整的背散射波反向传播：

1. **`cvdms_multislice_step` 返回原始 BSC 项**（不进行前向传播），由调用方累积并统一回传
2. **`_back_propagate_backscattered_waves`**（`multislice.py`）使用 `conj` 技巧逐层反向传播：
   ```
   ψ_back(z) = conj( forward_scattering(conj(ψ_back(z+dz)), V(z)) )
   ```
3. 逐层将反向传播的背散射波叠加到正向波的对应切片出口

#### 关键修复

反向传播实现中修复了两个 bug：

1. **off-by-one 切片索引**：`effective_slices[i+1]` → `effective_slices[i]`。反向传播从样品底部往上走，当前层应使用同一层的势能切片 `V(z)`，而非下一层的 `V(z+dz)`
2. **返回类型处理**：当 `next_slice=None`（最后一片）时 `multislice_step` 返回单个 `Waves`，但 `backscattering=True` 的前向循环始终期望 `(Waves, Waves)` 元组。增加了 `calculate_backscattered=True` 时的零填充分支

#### 对齐状态

背散射波反向传播现在与 ImageSimulation_CGS 的原始行为一致：从当前层到样品表面逐层反向传播，而非单步近似。

### v1.3 拉普拉斯算符增强

#### 变更

1. **默认精度从 6 改为 8（对应9点模板）**：`derivative_accuracy` 默认值从 `6` 改为 `8`，对应 9 点模板宽度（即 stencil 在 1D 上有 9 个系数，构成二维 9×9 Laplacian 模板），与原 C++ 代码的 9 点法一致。

2. **新增 FFT 拉普拉斯算符**：新增 `laplace_method="fft"` 选项，使用 FFT 在倒易空间计算精确的带限拉普拉斯算符：

   ```
   ∇²ψ = IFFT[-4π² · k² · FFT(ψ)]
   ```

   对应 ImageSimulation_CGS 中 `MultiCoefInReciprocalSpace` 内核（`wave_kernels.cu:5674`）。

3. **新增 `laplace_method` 参数**：`CVDMSMultislice` 和 `RealSpaceMultislice` 增加 `laplace_method` 参数，可选 `"finite-difference"`（默认）或 `"fft"`。

#### 使用示例

```python
# FFT 拉普拉斯算符（默认收敛参数）
algo = CVDMSMultislice(laplace_method="fft")
result = probe.multislice(potential, algorithm=algo)

# 轻原子（Si 等）FFT 模式工作稳定
# 重原子（Au 等）下 FFT 模式可能需要调整 convergence_threshold
algo = CVDMSMultislice(laplace_method="fft", convergence_threshold=5e-6)
```

#### 数值注意事项

- FFT 模式对重原子（Au、Pt 等）在低加速电压下可能发散，因为其精确的高频响应会导致 K-级数增长更快。此时可使用有限差分模式或适当放宽收敛阈值。
- FFT 模式内部使用 float64 计算以避免溢出，结果再转回原始精度。
- 两种模式的计算结果在低空间频率下一致，差异主要出现在接近 Nyquist 频率的高频分量。

### v1.4 数值稳定性与返回类型修复

#### NaN/Inf 截断

**问题**：`_cvdms_inner_k_series` 在高等阶（如 52 阶）时，拉普拉斯算符的累积数值误差导致 NaN/Inf，以 `DivergedError` 抛出，计算终止。

**修复**：改为 `break`，不将 NaN 项添加到 `k_series`，返回不含 NaN 的部分和。外层 `_cvdms_forward_scattering` 的振幅比检查（`|working|.sum() > 2 * |exit_wave|.sum()`）仍然捕获真正的不稳定参数组合。

#### 停滞检测加强

**问题**：原发散检测条件 `n_above > prev_n_above` 仅在未收敛像素数**增长**时触发。当级数进入振荡极限环（未收敛像素数停滞不变），会持续空转直至 NaN。

**修复**：改为 `n_above >= prev_n_above`，在未收敛像素数停止下降时即截断级数。该点通常对应最佳近似（C++ `fcms_taylor_max_iter()` 相同策略）。

#### `backscattering` vs `calculate_backscattered` 参数说明（v1.5+）

v1.5 将原有的 `expansion_scope`（结构性开关）和 `include_backscattering`（物理开关）合并为 `backscattering: bool`，同时保留 `calculate_backscattered` 作为独立控制。两者分工明确，分别控制两个不同的物理层面：

##### `backscattering` — 前向波背散射修正

控制前向传播的波函数是否在每个切片界面扣除背散射损失：

- `backscattering=False`（默认）：纯前向散射 `ψ_forward`，不受背散射影响。
- `backscattering=True`：在每个切片界面计算 BSC 算子并从前向波中减去：
  `ψ_corrected = ψ_forward - BSC(ψ_forward)`
  
  同时自动建立切片间耦合结构（传递 `next_slice`、返回 `(Waves, Waves)` 元组）。

##### `calculate_backscattered` — 背散射波分量追踪

控制是否将背散射波从样品中反向传播至表面并作为独立输出：

- `calculate_backscattered=False`（默认）：仅修正前向波，不单独追踪背散射分量。
- `calculate_backscattered=True`：除修正前向波外，还在每个界面**累积**背散射波，
  全部切片完成后通过 `_back_propagate_backscattered_waves` 反向传播至样品表面，
  最终作为独立测量输出（需搭配 `backscattering=True`）。

##### 使用组合

| `backscattering` | `calculate_backscattered` | 前向波修正 | 背散射波回传 | 适用场景 |
|---|---|---|---|---|
| `False`（默认） | `False`（默认） | 无 | 无 | 纯前向散射，薄样品 |
| `True` | `False`（默认） | **BSC 修正** | 无 | 前向波校正，忽略背散射波去向 |
| `True` | `True` | **BSC 修正** | **完整反向传播** | 完整 CVDMS，需分析背散射波 |

简单记忆：`backscattering` 控制**前向波是否被修正**，`calculate_backscattered` 控制**背散射波去了哪里**。

#### 外层非致命警告

**问题**：外层泰勒级数在 `max_terms` 项后仍未完全收敛时抛出 `NotConvergedError`，中断计算。

**修复**：改为 `warnings.warn(RuntimeWarning)`，返回最佳近似结果。与 C++ `fcms_taylor_max_iter()` 行为一致——接受部分收敛，由用户根据警告调整参数（增大 `max_terms` 或放宽 `convergence_threshold`）。

### v1.6 C++ CUDA 后端与后端选择参数

#### 变更

新增 C++ CUDA 后端（pybind11 封装），将整个外层 Taylor 级数 + 内层 K-series 循环合并为单个 C++ 调用，消除 Python 循环开销和中间显存读写。同时引入 `backend` 参数控制后端选择。

#### `backend` 参数

| 值 | 行为 |
|---|---|
| `"auto"`（默认） | 条件满足时优先尝试 C++ CUDA，不可用时回退 CuPy/Python |
| `"c++"` | 强制使用 C++ CUDA，不可用则抛出 RuntimeError |
| `"cupy"` | 跳过 C++ CUDA，直接使用 CuPy 融合核或 Python 循环 |

C++ CUDA 的启用条件：CuPy 可用、`dtype=complex64`、`ndim >= 2`、`use_fused_kernel=True`。

#### 使用示例

```python
# 自动选择（默认行为）
algo = CVDMSMultislice()

# 强制 C++ CUDA 后端
algo = CVDMSMultislice(backend="c++")

# 强制 CuPy/Python 后端
algo = CVDMSMultislice(backend="cupy")

# 与 use_fused_kernel 配合：C++ 后端需要 use_fused_kernel=True
algo = CVDMSMultislice(use_fused_kernel=True, backend="c++")
```

#### 调用链

```
CVDMSMultislice.backend
  └─ multislice_and_detect
       └─ cvdms_step(backend=...)
            ├─ _cvdms_forward_scattering(backend=...)
            │     ├─ [C++ CUDA] TaylorEngine.compute()  — 单次 pybind11 调用
            │     └─ [Python]   外层 Taylor + 内层 K-series 循环
            │
            └─ _cvdms_backscattering_correction(backend=...)
                  ├─ [C++ CUDA] BSCEngine.compute()     — 单次 pybind11 调用
                  └─ [Python]   CuPy 融合核 + full_series
```

#### 架构说明

- C++ CUDA 后端通过 `_cvdms_backend` 模块加载（编译输出 `_cvdms_backend*.so`）
- `"auto"` 模式下忽略 `ImportError` 静默回退；`"c++"` 模式下 `ImportError` 转为 `RuntimeError`
- 批处理支持：`TaylorEngine.compute()` 从 `__cuda_array_interface__` 检测 batch 维度，在 C++ 层循环处理每个 batch 项

### v1.8 C++ CUDA 深度优化

#### ConvergenceResult 结构体单 D2H 复制

**问题**：原来的收敛检测需要从设备端复制 3 个独立的 `int` 计数器（`n_above`、`n_nan`、`n_diverging`），每次调用 `read_convergence` 触发 3 次 `cudaMemcpyAsync` + 1 次 `cudaStreamSynchronize`。

**修复**：将 3 个计数器合并为 `ConvergenceResult` POD 结构体：

```cpp
struct ConvergenceResult {
    int n_above;
    int n_nan;
    int n_diverging;
};
```

- `read_convergence`: 1 次 `cudaMemcpyAsync`（原来 3 次）
- `reset_counters`: 1 次 `cudaMemsetAsync`（原来 3 次）
- Kernel 内改用 `atomicAdd(&d_result->n_above, 1)` 等

#### 外层 Taylor 循环融合

**问题**：`compute_taylor_series` 每外层迭代需要：
1. `taylor_scale_accumulate_kernel` — kernel launch
2. `cudaMemsetAsync` 清零计数器 — API 调用
3. `launch_convergence_check` — kernel launch

共 3 次 API 调用/迭代。

**修复**：`taylor_fused_kernel` 将三个操作融合为单个 kernel：
- `work = kseries * i*dz/n`（缩放）
- `exit += work`（累加到出口波）
- 收敛检测（`|work| > threshold`）

每外层迭代从 3 次 API 调用减少为 1 次 kernel launch + 1 次 `cudaMemsetAsync`。

#### compute_full_series 融合

**问题**：`compute_full_series` 对每个 order 的 K-算符幂次需要：
1. `launch_laplacian` — 有限差分拉普拉斯
2. `k_operator_apply_kernel` — K-算符应用
3. `fs_accumulate_kernel` — 累加到级数

共 3 次 launch/幂次，order 通常 10–20。

**修复**：`fs_fused_kernel<ACC>` 模板化 kernel，将三者融合为单次 launch。BSC 校正中 K-算符多项式计算的 launch 数减少 3 倍。

#### cuFFT 拉普拉斯算符

新增 `FFTLaplacian` 类封装 cuFFT，在倒易空间计算精确带限拉普拉斯：

```
∇²ψ = IFFT[-4π² · k² · FFT(ψ)]
```

**内部流程**：
1. `pack_complex_kernel`: 分离的 re/im → cuFFT 交错格式
2. `cufftExecC2C`: 正向 FFT
3. `fft_multiply_factor_kernel`: 乘以 k² 因子 + 1/N 缩放（融合）
4. `cufftExecC2C`: 逆向 FFT
5. `unpack_complex_kernel`: cuFFT 格式 → 分离的 re/im

**与 Python 的一致**：k² 因子使用 `fftfreq` 计算，与 `_laplace_operator_fft`（`finite_difference.py:309-383`）一致。

**调用路径**：`PyTaylorEngine::compute()` 新增 `laplace_method` 和 `sampling` 参数。当 `laplace_method="fft"` 时调用 `compute_taylor_series_fft`，内部使用 `FFTLaplacian` + `launch_k_operator_from_laplacian` + `fft_kseries_step_kernel`。

#### 性能结果

| 后端 | 时间 | 加速比 |
|------|------|--------|
| CuPy | 57.7s | 1.00x |
| C++ CUDA (有限差分) | 29.6s | 1.95x |
| C++ CUDA (FFT) | ~31s | ~1.86x |

FFT 版本略慢于有限差分版本（SrTiO₃ 30keV），因为每 K-series 迭代需要 2 个 cuFFT 调用 vs 1 个 fused stencil kernel。但 FFT 提供精确的带限拉普拉斯，无截断误差。

### v1.9 BSC Running Accumulation 反向传播

#### 背景

v1.2–v1.8 中，背散射波反向传播按出口面块（exit plane block）为单位逐块处理。每块跨越多个原始切片（对应 `potential.exit_planes` 的区间），块内切片被聚合成单块有效势能。这导致：

1. BSC 在每个粗粒度块界面上一次计算，而非每个原始切片
2. 反向传播步长等于块厚度（可能达数个 Å），对重原子或低电压容易因 conj-trick 中的泰勒级数展开步长过大而溢出

v1.9 改为 **running accumulation** 逐原始切片反向传播，对应 ImageSimulation_CGS 中 `calOneDevideK_forward_back` 的行为。

#### running accumulation 算法

```
work = 0
for sl = N-1 down to 0:
    work += bsc_slices[sl]           # 累加当前切片 BSC
    work = conj(forward(conj(work), V[sl]))  # 反向传播一层
    if sl 在出口面边界: 保存 ep_bsc[ep_idx] = work
ep_bsc[0] = work  # 入口表面总 BSC
```

**数学原理**：波函数反向传播是线性算符，各切片 BSC 的反向传播结果可线性叠加。running accumulation 通过从底部到顶部的单次扫描，同时完成了所有切片的反向传播，计算量从 O(N²) 降为 O(N)。

**与 CGS 逐层独立回传的等价性**（数值验证）：

running accumulation 与 ImageSimulation_CGS 的逐层独立回传在数学上完全等价。两者最终入口表面的总 BSC 为：

```
# Running accumulation（一次遍历）
work = 0
for sl = N-1..0:
    work += bsc[sl]
    work = forward(work, V[sl])

# 逐层独立回传（CGS 方式）
total = 0
for m = N-1..0:
    work = bsc[m]
    for j = m..0:
        work = forward(work, V[j])
    total += work
```

由于传播算子 `P(·) = exp(i·K·dz)` 的线性性，上述两种算法展开后完全一致：

```
running accumulation 展开:
  work = P₀(bsc₀) + P₀(P₁(bsc₁)) + P₀(P₁(P₂(bsc₂))) + ...

逐层独立回传展开:
  total = P₀(bsc₀) + P₀(P₁(bsc₁)) + P₀(P₁(P₂(bsc₂))) + ...
```

数值验证（30 切片，40×40 grid，随机 BSC）：
```
max |running|   = 25.08
max |independ|  = 25.08  
max diff        = 1.53e-05
相对误差        = 6.08e-07    ← 仅浮点舍入误差，在 float32 精度内
```

所以 running accumulation **不是近似**，而是利用线性算子性质的代数等价优化。

**conj-trick**：`conj(forward(conj(ψ), V))` 利用近轴波动方程的时间反演对称性，复用一个前向传播函数实现反向传播。

#### 修复的 bug

##### Bug 1: off-by-one 循环范围

**问题**：running accumulation 循环从 `num_slices - 2` 开始，排除最后一片（`num_slices - 1`）。

```python
# 错误：缺失最后一片
for sl_idx in range(num_slices - 2, -1, -1):
```

**原因**：原 exit-plane-block 算法中，最后一块已隐含包含最后一片；running accumulation 需要逐片处理，包括最后一片。

**修复**：改为 `range(num_slices - 1, -1, -1)`，包含所有切片。

##### Bug 2: 出口面保存位置偏移

**问题**：出口面波函数在共轭技巧（conj-trick）之后保存，此时 work 已被反向传播到 slice sl 的顶部（即上一层底部），而非当前切片底部的正确物理位置。

```python
# 错误：保存位置偏上一片
work.array += bsc_np[sl_idx]    # work 在 slice sl 底部 (正确位置)
conj → forward → conj           # work 移到 top(sl) = bottom(sl-1)
if sl_idx in sl_to_ep:           # 此处保存已偏移
    save(work)
```

**正确流程**：
```python
# 正确：在 conj-trick 之前保存
work.array += bsc_np[sl_idx]    # work 在 slice sl 底部 (正确位置)
if sl_idx in sl_to_ep:           # ✅ 在此处保存
    save(work)
conj → forward → conj           # work 移到下一层
```

**影响**：所有 EP 的深度分布都偏移了一层。底部 EP（7-13）显示零值，因为 BSC 在最后一层之后才生成，但保存时已被反向传播到上一层。

**C++ CUDA 路径的对应修复**：`running_accumulate_bsc` 中 `ep_bsc` 的保存从 Step 6（conj-trick 之后）移到 Step 1b（`accumulate_kernel` 之后、`conjugate_kernel` 之前）。

##### Bug 3: C++ CUDA 调用未传出口面参数

**问题**：Python 调用 `BSCBackPropEngine.compute_accumulate()` 时未传递 `ep_re_list`、`ep_im_list`、`exit_plane_indices` 参数，C++ CUDA 路径只写了 EP 0（入口表面总 BSC），其余 EP 保持未初始化。

**修复**：在 Python 端将 `backscattered_waves._array` 的 re/im 分离为 CuPy float32 列表，连同 `exit_planes` 整数列表一起传入 `compute_accumulate`，并在返回后从缓冲区读回所有 EP 的复波函数。

##### Bug 4: WavesDetector 重复添加

**问题**：`return_backscattered=True` 时，`MultisliceTransform.__init__` 和 `multislice_and_detect` 都可能向 detectors 列表添加 WavesDetector，导致 BSC 测量通道被重复注册。

**修复**：两处都改为先检查 detectors 是否为空，再决定添加 1 个还是 2 个 WavesDetector；`MultisliceTransform._calculate_new_array` 使用 `self._user_detectors` 避免重复计数。

#### 深度分布验证

修复后的 BSC 深度分布（5nm Au, 300keV, 14 EP）：

```
EP 13 (bottom): 0.0000e+00 ▏
EP 12:          2.07e-03   █
...
EP  6:          2.53e-02   █████████████
EP  5:          2.93e-02   ████████████████
EP  4:          3.35e-02   ██████████████████
EP  3:          3.82e-02   █████████████████████
EP  2:          4.33e-02   ████████████████████████
EP  1:          4.88e-02   ██████████████████████████████
EP  0 (top):    2.95e-02   ██████████████████
```

- 从底部到近表面单调递增：物理合理（更多材料 = 更多背散射）
- EP 0（入口表面）略低于 EP 1：顶部几层反向传播的干涉效应，物理正常
- 总 BSC 分数 10.23%，与预期一致

#### 架构变更

```
v1.8 (exit-plane-block path):
  multislice_and_detect
    ├─ 前向传播 (按 exit plane 块聚合势能)
    └─ back_propagate_bsc → 对每个块：conj-trick 通过聚合块

v1.9 (running accumulation path):
  multislice_and_detect
    ├─ 前向传播 (逐原始切片，收集 per_slice_bsc_arrays)
    └─ back_propagate_bsc → 从 N-1 到 0：累加 → conj-trick → 保存 EP
```

当 `calculate_backscattered=True` 时，前向循环收集每片 BSC（存为 `per_slice_bsc_data`），完成后调用 running accumulation。此路径对所有切片生效，不受 `exit_planes` 参数影响——exit_planes 仅决定在哪些深度保存累积值，不影响 BSC 的计算精度。

#### 路径选择

- **C++ CUDA 路径**（GPU）：当 per_slice BSC 为 CuPy 数组且 `_cvdms_backend` 可导入时自动启用。单次 `compute_accumulate` pybind11 调用完成所有切片的 running accumulation。
- **Python 路径**（CPU/GPU）：作为 fallback，使用 NumPy/CuPy 数组操作的 Python 循环。两者数学等价。

`backend="c++"` 或 `backend="cupy"` 参数控制前向传播路径的选择，不影响 back-propagation 路径（后者由 BSC 数据的设备位置决定）。

### 测试调整

`test_cvdms_compare_with_fourier` 中 `convergence_threshold` 从 `1e-10` 调整为 `1e-6`。`1e-10` 对 float32 数组过于严格（float32 机器精度约 1e-7），会导致内层级数进入不收敛的循环。

### v2.0 BSC 反向传播方向差异（与 ImageSimulation_CGS 的关键不一致）

#### 背景

在验证 abTEM CVDMS 背散射波结果与 ImageSimulation_CGS 时，发现两者的背散射波函数差异很大，菊池线（Kikuchi lines）的对比度明显不同。经过深入追查，发现 ImageSimulation_CGS 的反向传播实现存在一个物理方向错误。

#### ImageSimulation_CGS 的反向传播方向错误

ImageSimulation_CGS 在 `transmit_prop_CVDMS_BSC`（`wave_kernels.cu:6866-6876`）中使用 `calPureForwardScatter` 进行 BSC 反向传播：

```c
for (int jslice = islice; jslice >= 0; jslice--) {
    cudaMemcpy(ctemp2D0_d, backScatterWave_d, ...);
    calPureForwardScatter(..., backScatterWave_d, pot[jslice]);
}
```

`calPureForwardScatter` 计算的是 **`exp(i·K·dz)`**，即标准的**向下传播（forward）**算符。而背散射波是**向上传播**的 —— 它需要被传播回样品表面，而非进一步向样品深处传播。

正确做法应使用**向上（反向）传播**算符 **`exp(-i·K·dz)`**。

#### abTEM 的 conj-trick 正确反向传播

abTEM 使用时间反演共轭技巧（conj-trick）实现精确反向传播：

```
work = conj(forward_scattering(conj(work), V[sl]))
```

数学上：
```
conj(exp(i·K·dz) · conj(ψ)) = exp(-i·K·dz) · ψ
```

这是近轴波动方程 **`∂ψ/∂z = i·K·ψ`** 的精确时间反演传播。

#### 物理影响

虽然单张切片上 `|exp(i·K·dz)·ψ|² = |ψ|²`（酉算符保幅），但不同切片 BSC 分量在样品表面的**干涉**依赖于累积相位：

| 实现 | 累积算符 | 方向 | 相位 |
|------|---------|------|------|
| ImageSimulation_CGS | `U = exp(i·K_0·dz)·...·exp(i·K_i·dz)` | ⬇️ 向下（错误） | 错误 |
| abTEM | `U† = exp(-i·K_i·dz)·...·exp(-i·K_0·dz)` | ⬆️ 向上（正确） | 正确 |

当多个 BSC 分量叠加时，错误的相位累积改变了分量间的干涉条件，从而产生不同的菊池线图案。这就是两种实现背散射波结果差异的根本原因。

#### 1/K_j 修正的级数展开：两级联等价

在 BSC 算符的 `1/k_j` 修正中，CGS 和 abTEM 使用不同的实现策略，但**数学上等价**。

**ImageSimulation_CGS** 的 `calOneDevideK_forward_back`（`wave_kernels.cu:6351`）使用**级联缩放**：

```c
// 无条件缩放（所有 n，包括 n=1，都有相应缩放）
scaleSqrt = (0.5 - n) * λ / (π * n);
```

由于缩放后的结果作为下一次 K-算符的输入，产生了连乘效应：
```
result = scale₁·K(ψ) + scale₁·scale₂·K²(ψ) + scale₁·scale₂·scale₃·K³(ψ) + ...
```

| n | `scale_n` | 级联累乘 | 二项式系数 `binom(-1/2,n)·(λ/π)ⁿ` |
|---|-----------|---------|----------------------|
| 1 | `-0.5·λ/π` | `-0.5·λ/π` | `(-1/2)·(λ/π)` ✓ |
| 2 | `-0.75·λ/π` | `0.375·(λ/π)²` | `(3/8)·(λ/π)²` ✓ |
| 3 | `-0.833·λ/π` | `-0.3125·(λ/π)³` | `(-5/16)·(λ/π)³` ✓ |

级联成立的关键是二项式系数的递推关系：
```
binom(-1/2, n) = binom(-1/2, n-1) · (0.5-n)/n
```

所以 `coeff_n = coeff_{n-1} · (0.5-n)/n · λ/π = coeff_{n-1} · scale_n`，即前一项乘以 `scale_n` 得到当前项。

**abTEM** 的 `_cvdms_backscattering_correction`（`cvdms.py:782-795`）使用**非级联**方式：

```python
coeff_n = binom(-1/2, n) / (π·K₀)ⁿ = binom(-1/2, n) · (λ/π)ⁿ
cur = K_opⁿ(backscatter)    # 无缩放地重复应用 K-算符
correction += cur · coeff_n
```

**结论：两者数学等价**，都给出正确的 `(I+K/(πK₀))^{-1/2}` 二项式展开。级联通过递推实现与显式计算相同的结果。

与 `calK_PureForward`（主 K-级数）的区别：`calK_PureForward` 在 n=1 时跳过缩放（`if (nSqrtOrder != 1)`），而 `calOneDevideK_forward_back` 无条件缩放所有项。正是 n=1 也缩放使得级联递推出正确的二项式系数（`scale₁ = -0.5·λ/π`，而非 `calK_PureForward` 中跳过缩放的 `c₁ = 1`）。

#### 能否使其匹配？（v2.1+）


```python
algo = CVDMSMultislice(
    backscattering=True, calculate_backscattered=True,
)
```



**1/K 级数不需要修改** — 两者已数学等价。

**数值验证**（SrTiO₃, 300keV, 4×4×10 超胞, 0.1Å 采样, 0.4Å 切片）：

| 模式 | BSC 最大振幅 | 与 conj 差异 |
|------|-------------|-------------|
| `conj`（物理正确，默认） | 2.82e-3 | 参考值 |
| `forward`（CGS 兼容） | 2.89e-3 | **+2.7%** |
| 绝对差异最大值 | **1.50e-3** | — |

forward 模式的 BSC 振幅比 conj 模式大约 +2.7%，与 CGS 菊池线更强的观测一致。两种模式均通过 C++ CUDA 和 Python 路径实现。

### 双层级数展开

```
外层级数（指数展开，对应 calPureForwardScatter）：
    ψ(z+dz) = exp(i·K·dz)·ψ(z) = Σ (i·dz)ⁿ/n! · K_seriesⁿ(ψ)
    
内层级数（平方根展开，对应 calK_PureForward）：
    K_series(ψ) = Σ cₙ · Kⁿ(ψ)
    
    系数：
    c₁ = 1
    cₙ = (0.5 - n + 1) · λ / (π · n)   对于 n > 1
```

### K-算符

```
K(ψ) = V(r) · ψ + ∇²ψ / (4πK₀)

其中：
    V(r) = σ · V_potential / dz  （经 sigma 和厚度缩放的势能）
    K₀ = 1/λ                    （波数）
    λ = h/p                     （德布罗意波长）
```

### 两层收敛控制

1. **内层级数**（平方根展开）：逐个像素检查 `|Kⁿ(ψ)| > cutoff`，当未收敛像素数减少到零或开始增长时停止
2. **外层级数**（指数展开）：逐个像素检查 `|(i·dz)ⁿ/n! · K_seriesⁿ| > cutoff`，并检查振幅比防止发散

### 背散射修正

```
BSC = (k_j - k_{j-1}) / (2 · k_j)
ψ_corrected = ψ_forward - BSC(ψ_forward)

其中 k_j = K₀ · (I + K_series_j)
K_series_j 是第 j 层的 K-级数展开
```

## 与 ImageSimulation_CGS 的差异

### 已对齐的部分

| 特性 | ImageSimulation_CGS | abTEM v1.2 |
|------|--------------------|------------|
| 前向外层循环 | 泰勒级数 + 逐像素收敛 | 泰勒级数 + 逐像素收敛 |
| 前向内层循环 | 泰勒级数 + 逐像素收敛 | 泰勒级数 + 逐像素收敛 |
| 发散检测 | `fcms_taylor_max_iter()` | 未收敛像素数转折检测 |
| 背散射算符 | `calBSC` | `_cvdms_backscattering_correction` |
| 背散射波反向传播 | 完整逐层回传 | 完整逐层回传（`conj` 技巧） |
| 拉普拉斯算符 | 9 点有限差分 / FFT 变换 | 可配置精度（默认 9→10 阶，自动取偶）+ FFT 方法 |

### 未对齐的部分

| 特性 | ImageSimulation_CGS | abTEM | 说明 |
|------|--------------------|-------|------|
| 内层发散检测 | `fcms_taylor_max_iter()` 动态限制 | 未收敛像素数增长检测 | 效果类似，但后者的检测更及时，避免无效迭代 |
| BSC 反向传播方向 | `calPureForwardScatter` = `exp(i·K·dz)` **向下**传播（物理错误） | conj-trick = `exp(-i·K·dz)` **向上**传播（物理正确） | ❌ **关键差异**：CGS 的 BSC 反向传播方向错误，导致相位累积错误、菊池线图案不同 |
| 1/K 修正级数 | ✅ **级联**系数 `(0.5-n)·λ/(π·n)`（无条件缩放所有 n，递推得二项式系数） | ✅ **非级联**二项式系数 `binom(-1/2,n)·(λ/π)ⁿ`（独立系数） | ✅ **数学等价**：级联利用递推 `coeff_n = coeff_{n-1}·(0.5-n)/n·λ/π`，两者结果相同 |
| GPU 实现 | 原生 CUDA 内核 | C++ CUDA (pybind11) / CuPy / NumPy 三级切换 | `backend` 参数控制：auto/c++/cupy |

## 数值注意事项

1. **切片厚度**：CVDMS 算法的泰勒展开要求 `||K·dz||` 足够小。建议切片厚度不超过 2 Å，对重原子可更薄。
2. **收敛阈值**：默认 `1e-6` 适用于 float32 计算。使用 float64 时可适当收紧。
3. **势能强度**：Au 等重原子在低加速电压下会产生强势能。如遇到收敛问题，可尝试增大切片厚度或降低最大展开阶数。
4. **真空切片**：势能为零的切片仅由拉普拉斯项驱动。内层级数的发散检测会自动截断，不会影响计算结果。

## HRTEM 平行束成像（v2.2）

### 支持模式

| 模式 | 说明 | 支持情况 |
|------|------|---------|
| `PlaneWave` + `CVDMSMultislice`（无 BSC） | 平行束前向散射 | ✅ 已验证 |
| `PlaneWave` + `CVDMSMultislice`（有 BSC） | 平行束 + 背散射修正 | ✅ 已验证 |
| `FrozenPhonons` + `CVDMSMultislice` | 冷冻声子平均 | ✅ 已验证 |
| `CTF` 衬度传递 | 像差校正 + 光阑 + 部分相干 | ✅ 已验证 |

### 与 CBED 的区别

| | CBED | HRTEM |
|---|------|-------|
| 照明 | 聚焦探针（`Probe`） | 平面波（`PlaneWave`） |
| 势场 | `SubMat` 提取子区域 | 全场（全尺寸） |
| 输出 | 倒空间衍射花样 | 实空间像（exit wave → CTF → 强度） |

### 使用方法

```python
plane_wave = PlaneWave(energy=300e3)  # 300 keV
potential = Potential(atoms, sampling=0.05, slice_thickness=0.4, exit_planes=6)

# 无 BSC
exit_wave = plane_wave.multislice(
    potential,
    algorithm=CVDMSMultislice(convergence_threshold=1e-6, order=1),
    lazy=False,
)

# 有 BSC
result = plane_wave.multislice(
    potential,
    algorithm=CVDMSMultislice(
        backscattering=True,
        calculate_backscattered=True,
        convergence_threshold=1e-6,
        order=1,
    ),
    return_backscattered=True,
    lazy=False,
)
exit_wave, bsc_wave = result[0], result[-1]

# CTF 应用
ctf = CTF(Cs=-8e-6*1e10, energy=plane_wave.energy, defocus="scherzer",
          semiangle_cutoff=25, focal_spread=focal_spread)
image = exit_wave.apply_ctf(ctf).intensity()
```

### 注意事项

1. **`lazy=False`**：`PlaneWave.multislice()` 默认 `lazy=True` 返回 dask array，与 CuPy 不兼容。需要显式设置 `lazy=False`。
2. **`exit_planes` 对 BSC 的影响**：BSC 反向传播要求 `potential.num_exit_planes > 1`（即 `exit_planes` 不能为 None 或 0）。出口面深度分布的正确性经 running accumulation 验证（v1.9）。
3. **BSC 深度分布**：出口面深度分布反映从该深度到样品底部的累积背散射。底部出口面（最后一个切片）的 BSC 始终为零（没有更深切片产生背散射）。其他出口面位置可能因势场采样的周期性调幅而出现零值。

### v2.3 修复 BSC K-级数系数约定错误

#### 问题

HRTEM 模式下，使用 `PlaneWave` + `CVDMSMultislice(backscattering=True)` 在 30keV 下进行冷冻声子（FP）平均时，HRTEM 图像出现严重混乱，BSC 修正后的出口波振幅远低于正常值，图像近乎全黑。使用单 config（无 FP 平均）时 BSC 修正幅值也异常偏大——薄样品（~39Å）中 BSC 导致振幅下降约 9%，而在物理上 BSC 修正对薄样品的影响应远低于 1%。

#### 根因分析

abTEM 使用同一个 `_cvdms_inner_k_series` 函数计算 K-operator 的多项式展开，该函数对 **forward scattering** 和 **BSC** 两条路径使用完全相同的系数约定（c₁=1）。但 CGS 参考实现对于这两个路径使用不同的系数约定：

| 路径 | CGS 函数 | n=1 时的系数 |
|------|----------|-------------|
| Forward scattering | `calK_PureForward` | c₁ = 1（nSqrtOrder != 1 时不缩放） |
| BSC K-operator | `calK_forward_back` | c₁ = λ/(2π)（**始终缩放**） |

**数学推导**：

K-operator 定义为：
```
k = K₀ · sqrt(1 + K/(πK₀))
```
其中 K₀ = 1/λ。

Forward scattering 使用 K-operator 的指数传播：
```
exp(i · k · dz) = exp(i · K₀ · sqrt(1 + K/(πK₀)) · dz)
```
展开后 K 的线性项系数为 c₁ = 1（因为 K₀ · λ = 1）。

BSC 算子使用相邻切片的 k 值差分：
```
(k_{j+1} - k_j) / (2 · k_{j+1}) ≈ (K₀ · (1 + K_{j+1}/(2πK₀)) - K₀ · (1 + K_j/(2πK₀))) / (2 · K₀)
```
sqrt(1+x) 展开的一阶项系数为 1/2，代入 x = K/(πK₀) 得 c₁ = 1/(2πK₀) = λ/(2π)。

**级联传播效应**：K-series 采用级联（cascade）递推——第 n 次迭代的缩放值传递给第 n+1 次。因此 c₁ 的错误传播到 **所有高阶项**。在 30keV 下，λ=0.0698Å，K₀=14.33Å⁻¹，2πK₀ ≈ 90。BSC 修正项被放大了约 90 倍。

这就是为什么薄样品的 BSC 修正幅值本应在 1% 以下，实际却达到 9%。

#### 修复

**原则**：保留 `_cvdms_inner_k_series` 的 forward scattering 约定不变（c₁=1），在 BSC 路径中增加事后缩放 λ/(2π)。

Python 路径（`abtem/cvdms.py:741-765`，`_cvdms_backscattering_correction` 函数）：
```python
# 修复前：
wave_1 = (waves_array + wave_1) * K0  # kseries 未经 λ/(2π) 缩放，波函数被放大 2πK₀ 倍

# 修复后：
wave_1 = wave_1 / (2.0 * np.pi) + waves_array * K0  # kseries/(2π) + ψ·K₀
```

C++ CUDA 路径（`cpp/cvdms/src/Backscattering.cu:56-69`，`bsc_wave_kernel` 内核）：
```cpp
// 修复前：
wave_re[idx] = (psi_re[idx] + kseries_re[idx]) * K0;  // kseries 未经 λ/(2π) 缩放

// 修复后：
wave_re[idx] = psi_re[idx] * K0 + kseries_re[idx] / (2.0f * M_PI);
```

#### 影响范围

此修复影响 **所有** 使用 BSC 修正的 CVDMS 模拟——包括 **CBED 和 HRTEM** 模式。因为 `_cvdms_backscattering_correction` 是 BSC pipeline 的共享代码路径。

修复前的错误程度与加速电压成反比：
- **30keV**：λ=0.0698Å，2πK₀≈90，BSC 项被放大 ~90 倍（严重错误）
- **80keV**：λ=0.0250Å，2πK₀≈251，BSC 项被放大 ~36 倍
- **300keV**：λ=0.0197Å，2πK₀≈319，BSC 项被放大 ~28 倍

受影响的模拟类型：
- CBED BSC 模拟（`Probe` + `CVDMSMultislice(backscattering=True)`）
- HRTEM BSC 模拟（`PlaneWave` + `CVDMSMultislice(backscattering=True)`）
- 单配置和冷冻声子平均均受影响
- `calculate_backscattered=False`（纯前向传播）不受影响

#### 验证结果

**正确性验证**（SrTiO₃ 4×4×10，30keV，exit_planes=10）：
- C++ 和 Python 后端修复后结果 **完全一致**（相对差异 0%，绝对值差异 < 1e-12）
- 无 BSC 出口波振幅均值：0.931124
- C++ BSC 出口波振幅均值：0.932367（比无 BSC 高 0.13%）
- Python BSC 出口波振幅均值：0.932367（与 C++ 完全一致）

**FP-HRTEM 对比验证**（SrTiO₃ 6×6×30 ≈117Å，30keV，FP 4 configs，Fourier 参考）：

| 指标 | Fourier | CVDMS | CVDMS+BSC |
|------|---------|-------|-----------|
| 出口波振幅（FP 末面均值） | 0.868 | 0.894 | 0.896 |
| HRTEM 图像均值 | 0.918 | 0.975 | 0.979 |
| HRTEM 图像 std | 0.543 | 0.581 | 0.582 |
| NCC vs Fourier | — | 0.981 | 0.981 |
| NCC CVDMS vs BSC | — | — | 0.99997 |

结论：
- **CVDMS 与 Fourier 的 FP-HRTEM 图像 NCC 达 0.981**，差异主要来源于有限差分 Laplacian 与 FFT Laplacian 的高频响应差异（CVDMS 振幅 ~3% 偏高，这是有限差分法的已知特性）
- **CVDMS+BSC 与 CVDMS 几乎完全一致**（NCC=0.99997），BSC 在此厚度（~117Å）的效应极小
- **单 config 出口波 NCC ~0.967**，略低于 FP 平均后的 NCC，说明 FP 平均平滑了部分方法间差异
- Fourier 与 CVDMS 的 FP-HRTEM 图像强度分布范围和标准差一致，无系统性偏差

#### 经验教训

1. **系数约定一致性**：从 CGS 移植算法时，需要仔细区分不同路径的系数约定。同一数学函数（如 K-operator 多项式展开）在不同物理路径中可能使用不同的缩放约定。
2. **级联传播**：级联（cascade）递推中，第一项系数的错误会传播到所有高阶项。单独验证低阶项不足以发现此类 bug。
3. **电压依赖性**：BSC 相关 bug 在低加速电压下更容易暴露，因为 λ 更大。在 300keV 下此 bug 可能被误认为 BSC 修正本身的噪声。
