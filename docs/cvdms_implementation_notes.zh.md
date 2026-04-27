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

### 测试调整

`test_cvdms_compare_with_fourier` 中 `convergence_threshold` 从 `1e-10` 调整为 `1e-6`。`1e-10` 对 float32 数组过于严格（float32 机器精度约 1e-7），会导致内层级数进入不收敛的循环。

## 算法结构

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
| GPU 实现 | 原生 CUDA 内核 | C++ CUDA (pybind11) / CuPy / NumPy 三级切换 | `backend` 参数控制：auto/c++/cupy |

## 数值注意事项

1. **切片厚度**：CVDMS 算法的泰勒展开要求 `||K·dz||` 足够小。建议切片厚度不超过 2 Å，对重原子可更薄。
2. **收敛阈值**：默认 `1e-6` 适用于 float32 计算。使用 float64 时可适当收紧。
3. **势能强度**：Au 等重原子在低加速电压下会产生强势能。如遇到收敛问题，可尝试增大切片厚度或降低最大展开阶数。
4. **真空切片**：势能为零的切片仅由拉普拉斯项驱动。内层级数的发散检测会自动截断，不会影响计算结果。
