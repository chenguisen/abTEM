# *ab*TEM: transmission electron microscopy from first principles

[![PyPI version](https://badge.fury.io/py/abTEM.svg)](https://badge.fury.io/py/abTEM)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
<!--- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jacobjma/abTEM/master?filepath=examples%2Findex.ipynb)--->
<!---[[[[![DOI](https://zenodo.org/badge/205110910.svg)g(https://zenodo.org/badge/latestdoi/2051109--->

[**Docs**](https://abtem.github.io/doc/intro.html)
| [**Install Guide**](https://abtem.github.io/doc/getting_started/install.html)
| [**Walkthrough**](https://abtem.github.io/doc/user_guide/walkthrough/walkthrough.html)
<!---| [**Examples**](https://github.com/jacobjma/abTEM/tree/master/examples)--->

*ab*TEM (pronounced "ab-tem", as in "*ab initio*") provides a Python API for running simulations of (scanning)
transmission electron microscopy images and diffraction patterns using the multislice or PRISM algorithms. It is
designed to closely integrate with atomistic simulations using the Atomic Simulation
Environment ([ASE](https://wiki.fysik.dtu.dk/ase/)), and to directly use *ab initio* electrostatic potentials from the
high-performance density functional theory code [GPAW](https://wiki.fysik.dtu.dk/gpaw/). *ab*TEM is open source, purely
written in Python, very fast, and extremely versatile and easy to extend.

## Installation

You can install *ab*TEM using `pip`:

```sh
$ pip install abtem
```

For detailed instructions on installing *ab*TEM,
see [the installation guide](https://abtem.github.io/doc/intro.html).

## Getting started

To get started using *ab*TEM, please visit
our [walkthrough](https://abtem.github.io/doc/user_guide/walkthrough/walkthrough.html).

<!---To try *ab*TEM in your web browser, please click on the following Binder link:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jacobjma/abTEM/master?filepath=examples%2Findex.ipynb)--->

## Practical guide to simulations

For a pedagogical guide into numerical transmission electron microscopy simulations with extensive use of *ab*TEM code, please see [this computational article](https://www.elementalmicroscopy.com/articles/EM000005).

## Citing *ab*TEM

If you find *ab*TEM useful in your research, please cite our methods article:

J. Madsen & T. Susi, "The abTEM code: transmission electron microscopy from first principles", Open Research Europe 1:
24 (2021), doi:[10.12688/openreseurope.13015.2](https://doi.org/10.12688/openreseurope.13015.2).

## Usage poll

Our code is used by so many people that it is hard to keep track. If you already use or are interested in using *ab*TEM, we'd appreciate if you could fill in our [poll](https://github.com/abTEM/abTEM/discussions/186)!

## Contact

* Write the [maintainer](https://github.com/tomasusi) or [lead developer](https://github.com/jacobjma) directly
* Bug reports and issues: [GitHub issues](https://github.com/abTEM/abTEM/issues)
* Discussion and development: [GitHub discussions](https://github.com/abTEM/abTEM/discussions)

Please send us bug reports, patches, code, ideas and questions.

---

## 🧬 CVDMS: Chen-Van Dyck Multislice (`feat/cgs_cvdms` 分支独有)

本分支在原始 abTEM 基础上实现了 **CVDMS（Chen-Van Dyck Multislice）高精度多层片算法**，含背散射修正（BSC）与全矫正特性。该实现从 ImageSimulation_CGS（C++/CUDA）移植而来，并做了理论改进。

### 算法结构

三层嵌套迭代：

```
外层 Taylor 展开:  exp(i·dz·K) = Σ (i·dz)ⁿ/n! · Kⁿ(ψ₀)       ← 指数传播子
  └─ 内层 K-series:  K_series(ψ) = Σ cₙ · Kⁿ(ψ)             ← 平方根展开
       └─ K-operator:  K(ψ) = V·ψ + ∇²ψ/(4πK₀)              ← 波矢算符
            └─ Laplacian:  有限差分 (8阶9点) / FFT
```

### 背散射修正 (BSC)

传统多层片法仅考虑前向散射。CVDMS 在每层界面引入背散射算子，从正向波中减去反射分量：

```
ψ_out = ψ_forward - BSC(ψ_forward)
```

#### 公式演进

| 方法 | 公式 | 问题 |
|------|------|------|
| **SBA**（原始 CVDMS） | B = (k_{j+1} − k_j) / (2·k_{j+1}) | 势能减小时 \|1−B\|² > 1，非幺正 |
| **Fresnel 通量守恒** ✅ | R = (k_j−k_{j+1})/(k_j+k_{j+1}), T = √(1−\|R\|²) | 概率流严格守恒 |

本分支用 **Fresnel 振幅反射公式** 替代了原始的 SBA 公式，确保：

- 前向透射振幅 T 保证 \|ψ_out\|² ≤ \|ψ\|²
- 消除虚假强度增益
- 参考: Micron 190 (2025) 103778

#### 累积背散射波

启用 `calculate_backscattered=True` 时，逐层保存 BSC 场，在完整前向扫描后通过时间反演（conj-trick）反向传播回样品入口面，得到物理累积背散射波。

### 全矫正 (Fully Corrected)

参数 `fully_corrected=True` 保证 BSC 路径的返回值类型一致性：

- 非末层: `(exit_wave, backscatter_field)` 二元组
- 末层（无下一层界面）: 强制返回 `(exit_wave, zero_backscatter)`，使调用方可无条件解包

### 实现文件

| 文件 | 说明 |
|------|------|
| `abtem/cvdms.py` | Python 主实现：K-series、BSC、Taylor 展开 |
| `abtem/cvdms_kernels.py` | CuPy fused CUDA 核（收敛检测、K-operator） |
| `abtem/multislice.py` | 集成入口 `multislice_and_detect()` + BSC 反向传播 |
| `cpp/cvdms/` | C++ CUDA 后端（BSCEngine、TaylorSeries、Laplacian 等） |
| `benchmarks/bench_cvdms*.py` | 性能基准 |
| `docs/cvdms_*.md/html` | 中英文设计文档、论文框架、公式推导 |

### 与上游 abTEM 的主要差异

| 维度 | 上游 abTEM (dev) | 本分支 (feat/cgs_cvdms) |
|------|------------------|------------------------|
| CVDMS 算法 | ❌ 无 | ✅ 完整实现 |
| BSC Fresnel 修正 | ❌ 无 | ✅ 通量守恒公式 |
| C++ CUDA 后端 | ❌ 无 | ✅ `cpp/cvdms/` |
| 论文框架 (中/英文) | ❌ 无 | ✅ `docs/cvdms_papers/` |

### 参考文献

1. J.H. Chen, D. Van Dyck, "Accurate multislice theory for elastic electron scattering in transmission electron microscopy" (1997)
2. Micron 190 (2025) 103778 — Flux-conserving Fresnel backscattering correction
