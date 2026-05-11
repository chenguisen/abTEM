# CVDMS C++/CUDA 后端

将 abTEM 的 CVDMS（耦合波动力学多切片）算法模块的核心计算从 Python/CuPy 重写为 C++/CUDA，通过 pybind11 绑定回 Python。

## 背景

### 为什么做 C++/CUDA 重写

原始 Python 实现有以下限制：
1. **Python 外层循环开销**：Taylor 级数的外层循环在 Python 中执行，每次迭代都有 Python 解释器开销
2. **收敛检测的 D2H 同步**：每 `check_interval` 次迭代都需要 `Device→Host` 同步读取收敛计数器，阻塞 GPU 管线
3. **kernel launch 延迟**：内层 K-series 每次迭代单独 launch kernel（K 算子 + accumulate）
4. **无法利用 fused kernel 优化**：外层 Taylor 缩放+累加在 Python 中分两步执行，额外访存

C++ 后端将整个算法（外层 Taylor + 内层 K-series + 收敛检测）融合为单个 pybind11 调用，所有循环在 GPU 侧完成。

### 设计原则

- **不照搬** `ImageSimulation_CGS` 的架构（全局单例、手动内存管理、单文件 11K 行）
- **全 RAII**：`DeviceArray<T>` 自动管理 `cudaMalloc/cudaFree`
- **无全局状态**：所有参数通过函数参数显式传递
- **保留 Python fallback**：不满足 C++ 条件时自动回退到 Python 路径

## 架构

```
abtem/cvdms.py
  │
  ├─ _cvdms_forward_scattering()
  │     ├─ [C++ backend] TaylorEngine.compute()        ← 本模块
  │     │     └─ compute_taylor_series()
  │     │           ├─ compute_k_series()               [内层 K-series]
  │     │           │     ├─ launch_k_operator()         [K(ψ) = V·ψ + ∇²ψ/4πK₀]
  │     │           │     │     ├─ launch_laplacian()    [9 点 Laplacian]
  │     │           │     │     └─ k_operator_apply()
  │     │           │     ├─ kseries_accumulate_kernel() [系数级联 + 累加]
  │     │           │     └─ launch_convergence_check()  [GPU atomic 收敛检测]
  │     │           └─ taylor_scale_accumulate_kernel()  [缩放 + 累加 + 收敛检测]
  │     │
  │     └─ [Python fallback] Python outer loop + cuDVM fused kernels
  │
  ├─ _cvdms_inner_k_series()        [未被 C++ 覆盖时 Python 内层 K-series]
  │
  └─ _cvdms_backscattering_correction()
        ├─ [C++ backend] BSCEngine.compute()         ← 本模块
        │     ├─ compute_k_series(psi, V_current)    [stream 1]
        │     ├─ compute_k_series(psi, V_next)        [stream 2]
        │     ├─ wave = K0 * (psi + kseries)
        │     ├─ compute_full_series()                [1/k 校正]
        │     └─ backscatter *= (1 + correction) / (2*K0)
        │
        └─ [Python fallback] 原 Python BSC 路径
```

### 调用链（C++ 路径）

```
用户代码
  → wave.multislice(potential, algorithm=CVDMSMultislice())
    → abTEM multislice pipeline
      → cvdms_multislice_step()
        → _cvdms_forward_scattering()          # cvdms.py
          → TaylorEngine.compute()             # module.cpp (pybind11)
            → compute_taylor_series()           # TaylorSeries.cu
              → (在 GPU 上循环 max_terms 次)
                  → compute_k_series()           # KSeries.cu
                  → taylor_scale_accumulate()    # 缩放 + 累加 + 收敛检测

        → _cvdms_backscattering_correction()    # cvdms.py (if backscattering=True)
          → BSCEngine.compute()                  # module.cpp (pybind11)
            → apply_backscattering()             # Backscattering.cu
              ├→ compute_k_series(stream 1)      # V_current
              ├→ compute_k_series(stream 2)      # V_next
              ├→ bsc_wave_kernel + diff
              ├→ compute_full_series()           # 1/k 校正
              └→ bsc_correct_kernel              # 最终缩放
```

### C++ 后端激活条件

所有条件同时满足时自动启用，否则静默回退到 Python 路径：

| 条件 | 检查位置 |
|------|----------|
| `use_fused_kernel=True` | `_cvdms_forward_scattering` 参数 |
| CuPy 数组 (`xp.__name__ == "cupy"`) | `get_array_module()` |
| `complex64` dtype | `waves_array.dtype == np.complex64` |
| `ndim == 2` | `waves_array.ndim == 2` |
| `_cvdms_backend` 可导入 | `try: from _cvdms_backend import TaylorEngine` |

## 文件结构

```
cpp/cvdms/
├── CMakeLists.txt              # 构建系统 (pybind11 + CUDA)
├── README.md                   # 本文档
├── include/cvdms/
│   ├── Array.h                 # DeviceArray<T> RAII 包装
│   ├── Convergence.h           # 收敛检测接口
│   ├── KOperator.h             # K 算子接口
│   ├── KSeries.h               # K-series 接口
│   ├── Laplacian.h             # Laplacian 接口
│   ├── TaylorSeries.h          # Taylor 级数接口
│   └── Backscattering.h        # BSC 接口（存根）
├── src/
│   ├── Convergence.cu          # GPU atomic 收敛检测
│   ├── Laplacian.cu            # 9 点紧凑 Laplacian
│   ├── KOperator.cu            # K(ψ) = V·ψ + ∇²ψ/4πK₀
│   ├── KSeries.cu              # 内层 K-series Σ cₙ·Kⁿ(ψ)
│   ├── TaylorSeries.cu         # 外层 Taylor exp(i·K·dz)
│   └── Backscattering.cu       # BSC 存根
└── bindings/
    └── module.cpp              # PYBIND11_MODULE(_cvdms_backend)
```

## 算法细节

### 9 点紧凑 Laplacian

```cuda
// 模板系数（×1/(6·dx·dy)）：
//       1    4    1
//       4  -20    4
//       1    4    1
//
// ∇²ψ[i,j] ≈ 1/(6h²) · [4·sum(4-邻域) + sum(对角线) - 20·中心]
```

- O(h⁴) 精度
- 周期性边界条件
- 相比 5 点模板：
  - 更各向同性（对角耦合改善了旋转不变性）
  - 额外访存可忽略（仅多读 4 个对角像素）

### K 算子

```
K(ψ) = V·ψ + ∇²ψ / (4πK₀)

其中:
  K₀ = 1/λ           (波数)
  V  = σ·V_proj      (投影势 × 相互作用常数)
  λ  = 电子波长
```

两步计算（在同一 CUDA stream 中：
1. `launch_laplacian()` → 计算 ∇²ψ 到临时 buffer
2. `k_operator_apply_kernel()` → 计算 K(ψ)

### 内层 K-series（平方根展开）

```
K_series(ψ) = Σ cₙ · Kⁿ(ψ)

系数:
  c₁ = 1
  cₙ = (0.5 - n + 1) · λ / (π · n)   对于 n > 1

级联方式（复用已验证的行为）：
  cur = ψ (初始值)
  for n = 1..max_inner:
    buf = K_operator(cur)       # cur 和 buf 不同 buffer，无别名
    kseries += coeff · buf      # 累加到结果
    cur = coeff · buf           # 系数级联用于下次迭代
    
收敛检测（每次迭代，GPU atomic）：
  n_above = |buf|² > threshold² 的像素数
  if n_above == 0 → 收敛
  if n_nan > 0   → 溢出
```

### 外层 Taylor 级数（指数展开）

```
exp(i·K·dz) = Σ (i·dz)ⁿ/n! · Kⁿ

实现：
  exit = ψ, work = ψ
  for n = 1..max_terms:
    kseries = K_series(work)     # 内层 K-series
    work = kseries · i · dz / n  # Taylor 缩放
    exit += work                 # 累加到 exit wave
    if |work| 全部 < threshold → 收敛
    if NaN 出现                → 溢出
    
收敛检测（每次迭代）：
  launch_convergence_check() on work buffer
  single D2H sync 读取三个 int 计数器
```

### 收敛检测（GPU atomic）

```cuda
__global__ void convergence_check_kernel(re, im, threshold, ...)
  每个线程：
    mag2 = re² + im²
    if isnan(mag2) or isinf(mag2):
      atomicAdd(count_nan, 1)
    elif mag2 > threshold²:
      atomicAdd(count_above, 1)
    if prev exists and mag2 > prev_mag2 * 1.5:
      atomicAdd(count_diverging, 1)
```

- 全部在 GPU 侧完成，无中间 D2H 同步
- 单次 `cudaMemcpyAsync` + `cudaStreamSynchronize` 读取三个计数器
- 每次 Taylor 迭代检测一次

## 与 Python 路径的关键区别

| 方面 | Python (use_fused_kernel=True) | C++ 后端 |
|------|-------------------------------|----------|
| 外层循环 | Python `for` 循环 | GPU `for` 循环 |
| 收敛检测 | 每 `check_interval=2` 次迭代 | 每次迭代 |
| Laplacian | 4 阶可分离（有限差分系数） | 9 点紧凑模板 |
| K 算子 | 融合在 K-series kernel 中 | 独立 launch（Laplacian + apply） |
| D2H 同步 | 每次收敛检测 1 次 | 每次 Taylor 迭代 1 次 |
| 内存管理 | CuPy GC | RAII DeviceArray |

## 构建

### 手动构建

```bash
cd cpp
mkdir -p build && cd build
cmake ../cvdms \
  -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
make -j$(nproc)
cp _cvdms_backend*.so /path/to/site-packages/
```

### 通过 pip 构建

```bash
pip install -e ".[gpu]"
```

`pyproject.toml` 中的 `[tool.scikit-build]` 配置会触发 CMake 构建 `cpp/cvdms/`。

### 编译要求

| 工具 | 版本 |
|------|------|
| CUDA | ≥ 11.0（测试于 12.0） |
| GCC  | ≤ 12（CUDA 12.0 不兼容 GCC 13） |
| CMake | ≥ 3.21 |
| pybind11 | ≥ 3.0 |
| Python | ≥ 3.10 |

## Python 绑定

### module.cpp — PYBIND11_MODULE

```cpp
PYBIND11_MODULE(_cvdms_backend, m) {
    py::class_<cvdms::PyTaylorEngine>(m, "TaylorEngine")
        .def(py::init<>())
        .def("compute", &cvdms::PyTaylorEngine::compute, ...);
}
```

### CuPy 设备指针提取

通过 `__cuda_array_interface__` 协议获取 CuPy 数组的设备指针：

```cpp
static float *get_device_ptr(py::handle obj) {
    auto cai = obj.attr("__cuda_array_interface__").cast<py::dict>();
    auto data = cai["data"].cast<py::tuple>();
    uintptr_t ptr = data[0].cast<uintptr_t>();
    return reinterpret_cast<float *>(ptr);
}
```

### Lazy buffer 分配

`PyTaylorEngine` 在首次调用 `compute()` 时分配 GPU 工作 buffer，跨多次调用复用：

```cpp
void initialize(std::size_t nx, std::size_t ny) {
    if (initialized_ && nx == nx_ && ny == ny_)
        return;  // 复用已有 buffer
    // 释放旧 buffer，分配新 buffer
    ...
}
```

## 数值验证

| 测试 | C++ vs Python 差异 | 说明 |
|------|--------------------|------|
| 均匀波 + 均匀势 | 0.00e+00 | 精确匹配 |
| 高斯波包 + 随机势 | ~1e-7 | float32 精度级 |
| 周期平面波 + 零势 | ~1e-6 | 强度守恒 |
| 随机波 + 随机势 | ~2e-4 | 不同 Laplacian 模板的离散化差异 |
| pytest (16 tests) | 全部通过 | CPU 路径回退测试 |

## 维护指南

### 添加新功能

1. 在 `include/cvdms/` 添加头文件
2. 在 `src/` 添加 `.cu` 实现
3. 在 `bindings/module.cpp` 添加 pybind11 绑定
4. 在 `CMakeLists.txt` 的 `CVDMS_SOURCES` 列表中添加文件

### 修改 Laplacian 模板

编辑 `Laplacian.cu` 中的 `laplacian_kernel_9pt`。若需要可分离模板（匹配 `finite_difference_coefficients`），参考 `git log` 中的历史版本。

### 调优性能

- block size: 当前 `(16, 16)` 的 2D block，可根据 GPU 架构调整
- K 算子：当前两步（Laplacian + apply）可 fused 为单 kernel 以节省显存带宽
- Stream: 所有 kernel 接受 `cudaStream_t` 参数，可多 stream 并行

## 当前能力

| 功能 | 说明 |
|------|------|
| 前向散射 Taylor 级数 | C++ CUDA 加速，完整 Taylor + K-series 循环在 GPU 侧完成 |
| 后向散射（BSC）修正 | C++ CUDA 双流并行版本，与 Python 路径精度完全一致 |
| 批处理（frozen phonon） | 支持 `(batch, nx, ny)` 3D 输入，C++ 自动按 batch 循环 |
| 收敛检测 | GPU atomic 每次迭代检测，单次 D2H 同步读取三计数器 |
| 9 点紧凑 Laplacian | O(h⁴) 精度，各向同性优于 5 点模板 |
| float32 (complex64) | 当前精度，CUDA 原生支持，性能最优 |

## 已知问题

- 仅支持 `float32` 精度（`complex64`）。`complex128` 路径使用 Python fallback
- Laplacian 假设 `dx = dy`（方形像素），非方形采样精度下降
