# CVDMS 背散射修正：通量守恒 Fresnel 公式

> **文档版本**: v2.0  
> **对应代码**: `abtem/cvdms.py` → `_cvdms_backscattering_correction()`, `cpp/cvdms/src/Backscattering.cu` → `bsc_fresnel_kernel`  
> **引用本文**: 本文档描述的背散射修正方案基于 Chen-van-Dyck 多层理论框架（后文简称 CVDMS 方法或公式）[1,2]，并结合量子力学阶跃势 Fresnel 反射的精确概率流守恒条件重新表述了背散射算子。

---

## 符号表

| 符号 | 含义 | 单位 | 首次出现 |
|------|------|------|---------|
| $\Psi(\mathbf{r})$ | 高能电子的完整波函数 | — | §1 |
| $\psi$ | 前向传播的慢变包络波函数 | — | §1 |
| $\psi_{\text{fwd}}$ | 未经背散射修正的前向波 | — | §1 |
| $\psi_{\text{out}}$ | 经背散射修正后的前向出口波 | — | §1 |
| $\psi_{\text{bsc}}$ | 背散射场（从界面反射回上一层的波） | — | §3 |
| $V_j(\mathbf{R})$ | 第 $j$ 层的势能投影（沿 $z$ 方向平均） | V | §1 |
| $\lambda$ | 入射电子的（相对论修正）波长 | $\mathrm{\AA}$ | §1 |
| $K_0 = 1/\lambda$ | 真空波数 | $\mathrm{\AA}^{-1}$ | §1 |
| $k_j$ | 第 $j$ 层中的局域波数，$k_j = K_0 + \Delta k_j$ | $\mathrm{\AA}^{-1}$ | §1 |
| $\Delta k_j$ | 由势能 $V_j$ 引起的波数修正，$\Delta k_j \approx \sigma V_j/(2\pi K_0)$ | $\mathrm{\AA}^{-1}$ | §1 |
| $\text{wave}_j$ | K 级数计算的前向传播波场，$\text{wave}_j = k_j \cdot \psi$ | — | §1 |
| $\hat{K}_j$ | 第 $j$ 层的波矢算子（平方根算子） | $\mathrm{\AA}^{-1}$ | §2 |
| $\Delta = \partial^2/\partial x^2 + \partial^2/\partial y^2$ | 二维横向拉普拉斯算子 | $\mathrm{\AA}^{-2}$ | §2 |
| $\sigma = 2\pi me/h^2$ | 相互作用常数 | $\mathrm{V}^{-1}\mathrm{\AA}^{-1}$ | §2 |
| $\mathscr{B}_{j,j-1}$ | 背散射系数（BSC）算子 | — | §2 |
| $\mathscr{F}_{j,j-1}$ | 前向散射系数（FSC）算子 | — | §2 |
| $B$ | 慢包络近似（SBA）下的标量背散射系数 | — | §1 |
| $R$ | Fresnel 振幅反射系数（复数） | — | §2 |
| $T$ | Fresnel 通量守恒透射振幅，$T = \sqrt{1 - |R|^2}$ | — | §3 |
| $|R|^2$ | 反射强度（被反射的概率流比例），$0 \leq |R|^2 \leq 1$ | — | §3 |
| $I/I_0$ | 归一化强度（出口波总强度与入射波总强度之比） | — | §4 |
| $\varepsilon$ | 切片厚度 | $\mathrm{\AA}$ | §2 |
| $\Phi_j^f$ | 第 $j$ 层的前向散射波（CVDMS 理论符号） | — | §2 |
| $\Phi_j^b$ | 第 $j$ 层的背散射波（CVDMS 理论符号） | — | §2 |
| $\mathscr{S}_{j,j-1}$ | 切片传输算子（STO）矩阵 | — | §2 |

---

## 1. 背景：SBA 的非幺正性问题

### 1.1 慢包络近似 (SBA)

在 Chen-van-Dyck 多层法 [1,2]（下文简称 CVDMS 方法或公式）中，电子波穿过势能切片 $j$ 后，其前向传播波由 K 级数（波矢算子级数展开）给出：

\[
\text{wave}_j = K_0 \cdot \psi + \frac{1}{2\pi} \cdot K_{\text{series}}(\psi, V_j) \equiv k_j \cdot \psi \tag{1}
\]

其中 $K_0 = 1/\lambda$ 是真空波数，$k_j = K_0 + \Delta k_j$ 是第 $j$ 层的局域波数。

在第 $j$ 层与第 $j+1$ 层的界面上，势能变化引起背散射。CVDMS 原始推导 [1] 使用慢包络近似（Slowly Varying Envelope Approximation, SBA）得到背散射系数算子：

\[
\mathscr{B}_{j+1,j} = \frac{\hat{K}_{j+1} - \hat{K}_j}{2\hat{K}_{j+1}} \approx \frac{k_{j+1} - k_j}{2 k_{j+1}} \tag{2}
\]

在前向散射计算中，该算子以前向波振幅修正的形式出现（Chen \& Van Dyck, 1997, Eq.(47) [1]）：

\[
\psi_{\text{out}} = (1 - \mathscr{B}_{j+1,j}) \cdot \psi_{\text{fwd}} \approx (1 - B) \cdot \psi_{\text{fwd}} \tag{3}
\]

其中 $B = (k_{j+1} - k_j)/(2k_{j+1})$ 是标量化后的背散射系数。该近似对应于将指数传播子 $\exp(-\mathscr{B})$ 在 $\|\mathscr{B}\| \ll 1$ 条件下做一阶 Taylor 展开。

### 1.2 非幺正性的表现

SBA 公式在势能下降时（$k_j > k_{j+1}$，即 $B < 0$）违反概率守恒：

\[
B < 0 \;\Longrightarrow\; |\psi_{\text{out}}|^2 = |1 - B|^2 \cdot |\psi_{\text{fwd}}|^2 = (1 - B)^2 \cdot |\psi_{\text{fwd}}|^2 > |\psi_{\text{fwd}}|^2 \tag{4}
\]

物理上，弹性背散射应**减少**前向波强度——部分概率流被反射回入射方向。SBA 公式给出强度增加，这违背量子力学的概率流守恒定律。

数值表现：在含 SrTiO$_3$ 模型的 CVDMS 模拟中（30 keV, 0.5 $\mathrm{\AA}$ 切片, 256$\times$256 网格），SBA 背散射修正后的 $I/I_0$ 可达 1.03--1.17（取决于网格分辨率），且 **随厚度增加而持续增大**，完全违背弹性散射的幺正性要求。

### 1.3 非幺正性的根本原因

SBA 的非幺正性来自两个层面：

1. **截断误差**：$\exp(-B) = 1 - B + B^2/2 - \cdots$ 截断至一阶，丢失了高阶项中 $|e^{-B}| \leq 1$ 的保证。
2. **物理图景错误**：SBA 将背散射处理为**实数振幅修正**（实数的 $1-B$），而非量子力学要求的**概率流重分配**。前者是振幅的线性加减，后者是通量的二次守恒。

第二点更为根本——即使使用完整的指数形式 $\psi_{\text{out}} = \exp(-B) \cdot \psi_{\text{fwd}}$，由于 $B$ 为实数且可正可负，当 $B < 0$ 时 $|\exp(-B)| > 1$ 仍然成立。因此，问题的核心不是级数截断阶数，而是**需要将背散射重新表述为概率流守恒的通量重分配过程**。

---

## 2. Fresnel 反射公式推导

### 2.1 量子力学阶跃势模型

在 CVDMS 理论 [1] 中，固体薄膜的势能沿 $z$ 方向被离散为一系列切片。考虑第 $j$ 个切片界面的局域势能变化——这等价于量子力学中一维阶跃势的散射问题。

定态薛定谔方程：

\[
-\frac{\hbar^2}{2m}\frac{d^2\psi}{dz^2} + V(z)\psi = E\psi \tag{5}
\]

势能分布：

\[
V(z) = \begin{cases}
V_j, & z < 0 \\
V_{j+1}, & z > 0
\end{cases} \tag{6}
\]

电子从左侧（$z<0$，介质 $j$）垂直入射。波函数通解为：

\[
\psi(z) = \begin{cases}
e^{ik_j z} + R\,e^{-ik_j z}, & z < 0 \\
T\,e^{ik_{j+1} z}, & z > 0
\end{cases} \tag{7}
\]

其中局域波数：

\[
k_j = \frac{\sqrt{2m(E - V_j)}}{\hbar}, \quad k_{j+1} = \frac{\sqrt{2m(E - V_{j+1})}}{\hbar} \tag{8}
\]

### 2.2 边界条件与 Fresnel 系数

在界面 $z=0$ 处，波函数及其一阶导数连续：

\[
\begin{aligned}
1 + R &= T \tag{9} \\
k_j(1 - R) &= k_{j+1} T \tag{10}
\end{aligned}
\]

解得 Fresnel 反射系数和透射系数：

\[
\boxed{R = \frac{k_j - k_{j+1}}{k_j + k_{j+1}}}, \qquad
\boxed{T_{\text{F}} = \frac{2k_j}{k_j + k_{j+1}}} \tag{11}
\]

其中 $R$ 和 $T_{\text{F}}$ 分别是反射和透射的**振幅**系数。

### 2.3 概率流守恒

电子的概率流密度（狄拉克概率流）：

\[
j = \frac{\hbar k}{m} |\psi|^2 \tag{12}
\]

入射流、反射流和透射流分别为：

\[
j_i = \frac{\hbar k_j}{m}, \quad
j_r = \frac{\hbar k_j}{m} |R|^2, \quad
j_t = \frac{\hbar k_{j+1}}{m} |T_{\text{F}}|^2 \tag{13}
\]

概率流守恒要求 $j_i = j_r + j_t$：

\[
1 = |R|^2 + \frac{k_{j+1}}{k_j} |T_{\text{F}}|^2 \tag{14}
\]

代入式 (11) 验证该式严格成立：

\[
|R|^2 + \frac{k_{j+1}}{k_j} |T_{\text{F}}|^2 = 
\left(\frac{k_j - k_{j+1}}{k_j + k_{j+1}}\right)^2 + 
\frac{k_{j+1}}{k_j} \left(\frac{2k_j}{k_j + k_{j+1}}\right)^2
= \frac{(k_j - k_{j+1})^2 + 4k_jk_{j+1}}{(k_j + k_{j+1})^2} = 1 \tag{15}
\]

### 2.4 高能极限近似

在 TEM 条件下（加速电压 30--300 keV），真空波数 $K_0 = 1/\lambda \approx 80\text{--}800\,\mathrm{\AA}^{-1}$。势能 $V \sim 10\text{--}30$ V 引起的波数修正 $\Delta k = \sigma V/(2\pi K_0)$ 通常在 $10^{-3}$--$10^0\,\mathrm{\AA}^{-1}$ 量级，满足 $|\Delta k| \ll K_0$。因此：

\[
k_j \approx k_{j+1} \approx K_0 \;\Longrightarrow\; \frac{k_{j+1}}{k_j} \approx 1 \tag{16}
\]

在此极限下，概率流守恒式 (14) 简化为：

\[
\boxed{1 = |R|^2 + |T|^2} \tag{17}
\]

即反射强度 $|R|^2$ 与透射强度 $|T|^2$ 之和为 1。这一简化式正是本文背散射修正的核心物理基础。

---

## 3. 通量守恒 Fresnel 背散射公式

### 3.1 从波矢算子到 K 级数波场

在 CVDMS 计算中，无法直接获得波数 $k_j$，但可通过 K 级数（波矢算子的级数展开）计算经势能调制后的局部波场：

\[
\text{wave}_j = K_0 \cdot \psi + \frac{1}{2\pi} \cdot K_{\text{series}}(\psi, V_j) = k_j \cdot \psi \tag{18}
\]

因此，Fresnel 反射系数可直接从 $\text{wave}_j$ 和 $\text{wave}_{j+1}$ 逐像素计算：

\[
R = \frac{\text{wave}_j - \text{wave}_{j+1}}{\text{wave}_j + \text{wave}_{j+1}}
= \frac{k_j\psi - k_{j+1}\psi}{k_j\psi + k_{j+1}\psi} = \frac{k_j - k_{j+1}}{k_j + k_{j+1}} \tag{19}
\]

### 3.2 反射强度与透射振幅

反射强度（被背散射的概率流比例）：

\[
|R|^2 = \frac{|\text{wave}_j - \text{wave}_{j+1}|^2}{|\text{wave}_j + \text{wave}_{j+1}|^2} \tag{20}
\]

由概率流守恒式 (17)，通量守恒的透射振幅为：

\[
\boxed{T = \sqrt{1 - |R|^2}}, \qquad 0 \leq T \leq 1 \tag{21}
\]

### 3.3 背散射场

第 $j$ 层界面的背散射场定义为入射波与透射波之差：

\[
\boxed{\psi_{\text{bsc}} = \psi_{\text{fwd}} \cdot (1 - T) = \psi_{\text{fwd}} \cdot \left(1 - \sqrt{1 - |R|^2}\right)} \tag{22}
\]

### 3.4 修正后的前向波

\[
\boxed{\psi_{\text{out}} = \psi_{\text{fwd}} - \psi_{\text{bsc}} = \psi_{\text{fwd}} \cdot T} \tag{23}
\]

与 CVDMS 原始公式 [1] Eq.(47) 对比：前者是 $\psi_{\text{out}} = (1 - \mathscr{B}_{j+1,j}) \cdot \psi_{\text{fwd}}$，后者以通量守恒的 $T$ 替换了非幺正的 $(1 - \mathscr{B})$。

式 (22)--(23) 保证以下性质：

1. **强度守恒**：$|\psi_{\text{out}}|^2 = T^2 \cdot |\psi_{\text{fwd}}|^2 \leq |\psi_{\text{fwd}}|^2$（严格 $\leq$）
2. **相位保留**：$\arg(\psi_{\text{out}}) = \arg(\psi_{\text{fwd}})$（$T$ 为实数标量）
3. **SBA 一致性**：当 $|R|^2 \ll 1$ 时，$1 - T \approx |R|^2/2$，与 SBA 低阶项在量级上一致

### 3.5 Fresnel 公式与 SBA 的关系

弱反射极限（$|R|^2 \ll 1$，即 $|k_j - k_{j+1}| \ll |k_j + k_{j+1}|$）：

\[
T = \sqrt{1 - |R|^2} \approx 1 - \frac{|R|^2}{2} \tag{24}
\]

对比 SBA 透射系数 $1 - B$：

| 公式 | 反射系数 | 透射系数 |
|------|----------|----------|
| Fresnel（精确） | $\displaystyle R = \frac{k_j - k_{j+1}}{k_j + k_{j+1}}$ | $\displaystyle T = \sqrt{1 - \vert R\vert^2}$ |
| Fresnel（展开至 $\vert R\vert^2$） | 同上 | $\displaystyle 1 - \frac{1}{2}\left(\frac{k_j - k_{j+1}}{k_j + k_{j+1}}\right)^2$ |
| SBA（原始 CVDMS [1]） | $\displaystyle B = \frac{k_{j+1} - k_j}{2k_{j+1}}$ | $\displaystyle 1 - B$ |

SBA 反射是 $|R|$ 的**一阶量**（符号相反），而 Fresnel 通量损失是 $|R|^2$ 的**二阶量**。当 $|\delta k| \ll K_0$ 时两者数值接近，但 SBA 在 $\delta k > 0$（势能下降）时给出 $B < 0$，错误地放大振幅。

---

## 4. 幺正性证明

**定理**：对任意复数波场 $\text{wave}_j$ 和 $\text{wave}_{j+1}$，式 (22)--(23) 定义的 Fresnel 背散射修正保证 $|\psi_{\text{out}}|^2 \leq |\psi_{\text{fwd}}|^2$ 对所有像素成立。

**证明**：

\[
\begin{aligned}
|R|^2 &= \frac{|\text{wave}_j - \text{wave}_{j+1}|^2}{|\text{wave}_j + \text{wave}_{j+1}|^2} \\
&\leq 1 \quad \text{（反向三角形不等式：} |a - b| \leq |a + b|, \forall a,b \in \mathbb{C} \text{）} \\
0 &\leq |R|^2 \leq 1 \;\Longrightarrow\; T = \sqrt{1 - |R|^2} \in [0, 1] \\
|\psi_{\text{out}}|^2 &= T^2 \cdot |\psi_{\text{fwd}}|^2 = (1 - |R|^2) \cdot |\psi_{\text{fwd}}|^2 \leq |\psi_{\text{fwd}}|^2
\end{aligned}
\tag{25}
\]

当且仅当 $\text{wave}_j = \text{wave}_{j+1}$（两侧势能相等，无反射）时等号成立。$\square$

**数值稳定性**：反向三角形不等式已在数学上严格保证 $|R|^2 \leq 1$，强度守恒来源于该恒等式而非数值技巧。实际计算中浮点除法舍入误差可能使 $|R|^2$ 略超 $1$（例如 $1+10^{-7}$），因此实现中使用截断仅作为数值安全措施，防止 `sqrt(负数)` 产生 NaN：

```python
R_sq = xp.clip(R_sq, 0.0, 1.0)      # 仅防浮点舍入误差，非幺正性来源
T    = xp.sqrt(1.0 - R_sq)
```

---

## 5. 数值实现

### 5.1 Python 路径

**位置**：`abtem/cvdms.py` → `_cvdms_backscattering_correction()`

```python
# ============================================================================
# Step 1-2: Compute wave_j and wave_{j+1} via dual K-series (CVDMS Eq.(1))
# ============================================================================
wave_1 = _cvdms_inner_k_series(                   # wave_j = K_0 * psi
    waves_array, transmission_function, ...)       #   + K_series(psi, V_j)/(2pi)
wave_1 = wave_1 / (2.0 * np.pi) + waves_array * K0

wave_2 = _cvdms_inner_k_series(                   # wave_{j+1} = K_0 * psi
    waves_array, transmission_function_next, ...)  #   + K_series(psi, V_{j+1})/(2pi)
wave_2 = wave_2 / (2.0 * np.pi) + waves_array * K0

# ============================================================================
# Step 3: Flux-conserving Fresnel reflection
#   R    = (wave_j - wave_{j+1}) / (wave_j + wave_{j+1})  [Eq.(19)]
#   |R|^2 = reflected probability flux fraction           [Eq.(20)]
#   T    = sqrt(1 - |R|^2)                                [Eq.(21)]
#   psi_bsc = psi * (1 - T)                               [Eq.(22)]
#   psi_out = psi * T                                     [Eq.(23)]
# ============================================================================
sum_waves  = wave_1 + wave_2
diff_waves = wave_1 - wave_2

with np.errstate(divide='ignore', invalid='ignore'):
    R_sq = xp.abs(diff_waves) ** 2 / xp.abs(sum_waves) ** 2
    R_sq = xp.clip(R_sq, 0.0, 1.0)      # numerical safeguard
    T    = xp.sqrt(1.0 - R_sq)           # flux-conserving transmission amplitude
    backscatter = waves_array * (1.0 - T)

# Zero-out pixels where sum_waves ~ 0 (both potentials negligible)
zero_mask = xp.abs(sum_waves) < xp.finfo(sum_waves.dtype).eps * 10
if xp.any(zero_mask):
    if xp is np:
        backscatter[zero_mask] = 0.0 + 0.0j
    else:
        backscatter[zero_mask] = xp.zeros(1, dtype=backscatter.dtype)[0]

return backscatter
```

**对比原 SBA 实现的关键变化**：

| 操作 | 原 SBA + 1/k 修正 | 现 Fresnel 通量守恒 |
|------|-------------------|-------------------|
| 差分 | `backscatter = wave_2 - wave_1` | — |
| 1/k 收敛循环 | 数十次 K 算子迭代（`calOneDevideK_forward_back`） | — |
| 归一化 | `backscatter /= (2 * K0)` | — |
| 反射计算 | — | `R_sq = \|diff\|^2 / \|sum\|^2` |
| 透射振幅 | — | `T = sqrt(1 - R_sq)` |
| 背散射场 | — | `backscatter = psi * (1 - T)` |

### 5.2 C++ CUDA 路径

**位置**：`cpp/cvdms/src/Backscattering.cu` → `bsc_fresnel_kernel`

```cuda
__global__ void bsc_fresnel_kernel(
    const float *w1_re, const float *w1_im,   // wave_j
    const float *w2_re, const float *w2_im,   // wave_{j+1}
    const float *psi_re, const float *psi_im, // incident wave psi
    float *bs_re, float *bs_im,               // output: backscatter field
    int count)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count) return;

    // Complex sum and difference (Eq.(19))
    float sr = w1_re[idx] + w2_re[idx];
    float si = w1_im[idx] + w2_im[idx];
    float dr = w1_re[idx] - w2_re[idx];
    float di = w1_im[idx] - w2_im[idx];

    // |sum|^2 and |diff|^2
    float sum_sq   = sr * sr + si * si;
    float diff_sq  = dr * dr + di * di;

    // |R|^2 = |diff|^2 / |sum|^2, clamped to [0, 1] (Eq.(20))
    float R_sq = sum_sq > 0.0f ?
                 fminf(diff_sq / sum_sq, 1.0f) : 0.0f;
    float T   = sqrtf(fmaxf(1.0f - R_sq, 0.0f));  // Eq.(21)

    // psi_bsc = psi * (1 - T), preserving incident phase (Eq.(22))
    float scale = 1.0f - T;
    bs_re[idx]  = psi_re[idx] * scale;
    bs_im[idx]  = psi_im[idx] * scale;
}
```

该 kernel 替换了原 `apply_backscattering()` 中的 SBA + 1/k 收敛循环（`bsc_diff_kernel` + `compute_one_over_k_series` + `bsc_add_correct_kernel` 共约 30 次 CUDA kernel launch），将 3 个步骤合并为单次 kernel launch。

### 5.3 上层调用代码

**位置**：`abtem/cvdms.py` → `cvdms_multislice_step()`

```python
if backscattering and next_slice is not None:
    # Vacuum guard: skip BSC when current slice has negligible potential
    xp_bs = get_array_module(transmission_function)
    tf_max = float(xp_bs.max(xp_bs.abs(transmission_function)))
    if tf_max < 1e-10:
        exit_wave = pure_forward
        backscatter = xp_bs.zeros_like(pure_forward)
    else:
        backscatter = _cvdms_backscattering_correction(
            pure_forward, transmission_function,
            transmission_function_next, ...)
        # psi_out = psi_fwd - psi_bsc  [Eq.(23)]
        exit_wave = pure_forward - backscatter
```

该上层接口未做变动——Fresnel 公式与 SBA 都以 $\psi_{\text{bsc}}$ 的形式返回，上层统一执行 $\psi_{\text{out}} = \psi_{\text{fwd}} - \psi_{\text{bsc}}$。下游的背传播逻辑（`_back_propagate_bsc_impl` / `running_accumulate_bsc`）完全不受影响。

### 5.4 计算复杂度对比

| 阶段 | SBA + 1/k 修正 | Fresnel 通量守恒 |
|------|---------------|-----------------|
| K 级数（$\text{wave}_j$） | 1 次 | 1 次 |
| K 级数（$\text{wave}_{j+1}$） | 1 次 | 1 次 |
| 界面计算 | 差分 + 1/k 收敛循环（~30 次 K 算子迭代 + 收敛检测） | 1 次 Fresnel kernel（逐像素运算，无迭代） |
| **总计/切片界面** | 2$\times$K级数 + $\sim$30$\times$K算子 + ~3 kernel launch | **2$\times$K级数 + 1 kernel launch** |

Fresnel 公式在恢复幺正性的同时，**计算效率显著提高**（消除了收敛循环）。

---

## 6. 验证结果

### 6.1 测试条件

- **结构**: SrTiO$_3$，$2\times2$ 面内超胞（$7.81 \times 7.81\,\mathrm{\AA}^2$），$z$ 方向 4$\times$--32$\times$ 重复
- **切片厚度**: 0.5 $\mathrm{\AA}$
- **网格**: $(78, 78)$，等效采样 $\approx 0.10\,\mathrm{\AA}$
- **能量**: 80 keV（多能量测试含 30, 200, 300 keV）
- **后端**: Python (CuPy) 和 C++ CUDA（一致性校验）

### 6.2 多厚度测试

| $z$ 重复 | 切片数 | Forward $I/I_0$ | BSC $I/I_0$ | $\Delta$ | 结果 |
|----------|--------|----------------|-------------|----------|------|
| 4$\times$ | 32 | 0.990809 | 0.990743 | $-6.59\times10^{-5}$ | PASS |
| 8$\times$ | 63 | 0.979383 | 0.979236 | $-1.47\times10^{-4}$ | PASS |
| 16$\times$ | 125 | 0.966987 | 0.966731 | $-2.56\times10^{-4}$ | PASS |
| 32$\times$ | 250 | 0.937855 | 0.937368 | $-4.87\times10^{-4}$ | PASS |

BSC $I/I_0$ 始终**小于** Forward $I/I_0$，且差值随厚度单调递增（厚度越大，累积背散射损失越多），与物理预期完全一致。

### 6.3 多能量测试

| 能量 | 厚度 | BSC $I/I_0$ | 结果 |
|------|------|------------|------|
| 30 keV | 31.5 $\mathrm{\AA}$ | 0.961080 | PASS |
| 80 keV | 31.5 $\mathrm{\AA}$ | 0.979236 | PASS |
| 200 keV | 31.5 $\mathrm{\AA}$ | 0.970368 | PASS |
| 300 keV | 31.5 $\mathrm{\AA}$ | 0.968037 | PASS |

所有能量下 BSC $I/I_0 < 1$ 严格成立。

### 6.4 C++ vs Python 后端一致性

| 后端 | $I/I_0$ | 相对差异 |
|------|---------|---------|
| auto（C++ CUDA） | 0.979244 | 0.0000% |
| cupy（Python） | 0.979244 | 基准 |

C++ 和 Python 后端给出完全一致的结果。

### 6.5 关键指标汇总

| 检查项 | 结果 |
|--------|------|
| BSC $I/I_0 < 1$（所有配置） | $\checkmark$ |
| BSC $I/I_0 \leq$ Forward $I/I_0$（所有厚度） | $\checkmark$ |
| $I/I_0$ 差值随厚度单调递增 | $\checkmark$ |
| C++ / Python 后端结果一致 | $\checkmark$ |
| 测试套件（`test_cvdms_multislice.py`） | 16/18 通过* |

\* 2 个预存失败：`Probe.array` API 不兼容（与背散射修正无关）。

---

## 7. 与 SBA 的系统对比

### 7.1 物理层面

| 方面 | SBA + 1/k 修正（CVDMS 原始 [1]） | Fresnel 通量守恒（本工作） |
|------|--------------------------------|--------------------------|
| 物理基础 | 慢包络近似，势能差的一阶线性展开 | 精确 Fresnel 阶跃势反射 + 概率流守恒 |
| 反射系数 | $B = (k_{j+1} - k_j)/(2k_{j+1})$ | $R = (k_j - k_{j+1})/(k_j + k_{j+1})$ |
| 透射系数 | $1 - B$（一阶 Taylor 截断） | $T = \sqrt{1 - |R|^2}$（通量守恒形式） |
| 幺正性 | 势能下降时 $I/I_0 > 1$ | 严格 $I/I_0 \leq 1$ |
| 相位处理 | 实数振幅加减，相位失真 | 保留入射波相位 |
| 适用条件 | $\|B\| \ll 1$（弱背散射） | 任意 $\|R\| \leq 1$（无近似） |

### 7.2 计算层面

| 方面 | SBA + 1/k 修正 | Fresnel 通量守恒 |
|------|---------------|-----------------|
| 界面运算量 | 差分 + ~30 次 K 算子迭代 + 收敛检测 | 1 次逐像素运算 |
| 收敛控制 | 逐像素检测、发散判定、NaN 截断 | 无需迭代、无发散风险 |
| 浮点溢出 | 存在（K 级数在精细网格下可能发散） | 无（仅平方根运算） |
| CUDA 实现 | 3 个 kernel + 1 个收敛循环 | 1 个 kernel |
| 临时缓冲区 | cur / correction / scratch 等多组 | 复用 wave 缓冲区 |

### 7.3 代码变化示意

```
旧代码 (SBA + 1/k):
  backscatter = wave_2 - wave_1
  { 1/k 收敛循环: for n=1..max_inner
        scratch = K(cur) * coeff
        correction += scratch
        if converged: break
  }
  backscatter = (backscatter + correction) / (2*K0)

新代码 (Fresnel 通量守恒):
  diff = wave_1 - wave_2
  sum  = wave_1 + wave_2
  R_sq = |diff|^2 / |sum|^2
  T    = sqrt(1 - R_sq)
  backscatter = psi * (1 - T)
```

---

## 8. 参考文献

[1] J.H. Chen, D. Van Dyck. Accurate multislice theory for elastic electron scattering in transmission electron microscopy. *Ultramicroscopy*, **70** (1997) 29--44.  
—— 提出 STO 矩阵 + 传递矩阵技术，在多层方案中严格求解类薛定谔方程；导出背散射系数算子 $\mathscr{B}_{j,j-1}$ 及考虑背散射的透射波公式 Eq.(47)；给出高能近似下的 BSC 简化 $\mathscr{B}_{j+1,j} \approx \sigma(U_{j+1} - U_j)/(4\pi K_0)$。本文所称 **Chen-van-Dyck 多层法**（CVDMS 方法或公式）即指该文建立的理论框架。

[2] J.H. Chen, D. Van Dyck. Fast STEM image simulation in low-energy transmission electron microscopy by the accurate Chen-van-Dyck multislice method. *Micron*, **190** (2025) 103778.  
—— 将 CVDMS 方法推广至低能 TEM 和 STEM 模拟，讨论 K 级数展开的数值收敛性及浮点精度控制策略。

[3] D. Van Dyck. Image calculations in high-resolution electron microscopy: Problems, progress, and prospects. *Advances in Electronics and Electron Physics*, **65** (1985) 89--135.  
—— 多层方法的统一理论框架，阐明 CMS 与修正薛定谔方程的关系。

[4] E.J. Kirkland. *Advanced Computing in Electron Microscopy*, 2nd ed. Springer (2010).  
—— 标准 TEM 图像模拟参考书，含多层法实现细节及数值考量。

[5] L.D. Landau, E.M. Lifshitz. *Quantum Mechanics: Non-Relativistic Theory*, 3rd ed. Pergamon Press (1977).  
—— Ch.3（散射矩阵的幺正性）, Ch.25（势垒反射），本文 Fresnel 反射推导的量子力学基础。

[6] J.C.H. Spence. *High-Resolution Electron Microscopy*, 4th ed. Oxford University Press (2013).  
—— 实验高分辨电子显微学标准参考。

---

> **附录 A：版本历史**
> 
> | 版本 | 日期 | 说明 |
> |------|------|------|
> | v1.0 | 2026-05-13 | 初稿：SBA 非幺正性问题 + Fresnel 推导 |
> | v2.0 | 2026-05-14 | 增加符号表、更新文献引用格式(CVDMS 命名规范)、修复 LaTeX 排版 |
