# 透射电子显微镜中弹性电子散射的精确多层理论

**Accurate multislice theory for elastic electron scattering in transmission electron microscopy**

**作者**: Jiang Hua Chen, Dirk Van Dyck*
**单位**: EMAT, University of Antwerp (RUCA), Groenenborgerlaan 171, B-2020 Antwerp, Belgium
**发表**: Ultramicroscopy 70 (1997) 29-44
**收稿**: 1997年4月18日; 修订稿: 1997年7月16日

---

## 摘要

结合**切片传输算子矩阵**（slice-transmission-operator matrix）的概念与**传递矩阵技术**（transfer matrix technique），本文在多层法（multislice scheme）框架内严格求解了固体中弹性电子散射的完整薛定谔方程。这一新方法扩展了多层理论的适用范围。研究表明，多层理论在所有弹性电子散射情形中原则上可以达到与 Bloch 波理论和 Green 函数理论同等的精度。本文指出，在某些情况下（如高阶劳厄带 HOLZ 反射的计算），应对传统高能近似以及传统多层法（CMS）进行更精确的修正。本文推导出了一种新的多层公式用于计算透射波。该公式考虑了背散射效应，因此在大光束倾斜角下也能保持精确。同时表明，新的多层方法保留了 CMS 相对于 Bloch 波方法的所有优势。

**PACS**: 61.14.Dc; 61.16.Bg
**关键词**: 多层理论; HOLZ 效应; 背散射效应

---

## 1. 引言

高能电子显微镜（HEEM）的最新发展——高相干场发射电子源、电子能量过滤器（electron energy-filters）和慢扫描CCD（charge-coupled device）相机逐渐成为仪器的组成部分——使得获取可靠的定量实验数据成为可能，且在实验选择上具有很大的灵活性（HREM/STEM/焦像、ED、CBED、ptychography、能量选择等）。然而，实验数据的解释要求理论也必须足够精确，以实现与实验的定量匹配。

对于定量电子显微学，多重弹性电子散射是电子携带物体结构信息的主要相互作用过程。高能弹性电子散射的精确模拟需要使用多束动力学衍射理论。目前有两种主流理论广泛用于多束动力学计算：**Bloch 波理论**和**多层理论**。其他理论在特定的 HEED 情形下也有其便利之处。在所有现有理论中，Green 函数理论原则上最为精确，因为它给出了固体中高能弹性电子散射的完整类薛定谔方程的最一般解，无需任何近似假设。但由于 Born 级数收敛慢，它在计算 HEED 和成像时不太实用。此外，Van Dyck 也表明，利用散射矩阵方法可以严格求解完整薛定谔方程，但所得解过于复杂而难以数值计算。

Bloch 波理论（有时也称为 Bethe 理论），包括倒空间方法和实空间方法，一直是理解和计算 TEM 中动力学衍射效应的最基本理论。虽然大多数 Bloch 波方法仅用于计算前向散射波，但原则上背散射波也可以包含在 Bloch 波方法中。然而，该理论不便于处理缺陷结构，也不能包含上下表面效应，因为它基于完美周期势的假设。当涉及大量束斑时（例如 HRTEM 图像模拟），其计算效率不如多层法。

多层法最初由 Cowley 和 Moodie 从物理光学原理提出，在许多 HEED 和成像的动力学计算情形中一直是最有效的方法，特别是当用 FFT 技术加速时。多层法相对于 Bloch 波方法的另一个重要优势是它不需要势函数的周期性，因此可用于非周期结构。此外，多层法还可用于非弹性散射效应的计算。研究表明，所有多层法都可在统一框架下视为高能电子衍射的约化（或修正）薛定谔方程的积分方法。然而，该方程基于高能近似，其中**背散射效应被忽略了**。因此，当这些效应不可忽略时，现有多层法将不再有效。

众所周知，两种最流行的理论都不是在所有 HEED 情形中都适用。事实上，两种方法在处理 RHEED 图案和图像时都遇到了困难，因此需要为这类动力学计算开发其他方法。一般来说，当入射束大幅度偏离表面法线时，传统高能近似不再有效，背散射效应必须被考虑。RHEED 只是背散射效应占主导的极端情形。因此，随着越来越先进的实验技术出现，迫切需要更精确但高效的动力学理论（或模拟方法）。

**本文的目的**是扩展多层理论（或方法）的有效性，使其能够考虑背散射效应，同时保留传统多层概念的所有优势。

---

## 2. 多层方案中薛定谔方程的一般解

### 2.1 问题的建立

为了包含高能电子波穿过固体薄膜时的背散射效应（图 1a），需要求解"相对论修正"的类薛定谔方程，该方程从 HEED 的 Dirac 方程导出：

$$-\frac{h^2}{8\pi^2 m} \nabla^2 \Psi(\mathbf{r}) + eU(\mathbf{r})\Psi(\mathbf{r}) + \frac{h^2 K_0^2}{2m}\Psi(\mathbf{r}) = 0 \tag{1}$$

其中 $U(\mathbf{r})$ 是晶体势函数，$\Psi(\mathbf{r})$ 是高能电子的波函数，$\nabla^2$ 是三维拉普拉斯算子，$e$ 是电子电荷，$h$ 是普朗克常数，$m$ 和 $K_0$（$= 1/\lambda$，$\lambda$ 是波长）分别是入射电子的"相对论修正"质量与波矢。

**图 1**: 固体薄膜中弹性电子散射示意图: (a) 前向散射和背散射; (b) 多层方案。

### 2.2 单层解

现在在多层方案框架内求解方程 (1)，其中固体被视为一系列平行于表面的极薄切片的组合（图 1b），每个切片的势函数沿 z 轴平均：

$$O_j(\mathbf{R}) = \int_{(j-1)\varepsilon}^{j\varepsilon} U(\mathbf{r})\,\mathrm{d}z \tag{2}$$

其中 $\mathbf{R} = (x, y)$，$\varepsilon$ 是切片厚度。不失一般性，方程 (1) 可重写为：

$$\frac{\partial^2 \Psi(\mathbf{r})}{\partial z^2} + [2\pi\hat{K}(\mathbf{r})]^2 \Psi(\mathbf{r}) = 0 \tag{3}$$

其中波矢算子 $\hat{K}(\mathbf{r})$ 定义为：

$$\hat{K}(\mathbf{r}) = K_0 \sqrt{1 + \frac{\Delta}{(2\pi K_0)^2} + \frac{\sigma}{\pi K_0}U(\mathbf{r})} \tag{4}$$

其中 $\Delta = \partial^2/\partial x^2 + \partial^2/\partial y^2$ 是二维拉普拉斯算子，$\sigma = 2\pi me/h^2$ 是相互作用常数。

对于整个固体薄膜，无法找到方程 (2) 的简单解。但对于单个切片，例如切片 $j$，有：

$$\frac{\partial^2 \Psi_j(\mathbf{r})}{\partial z^2} + (2\pi\hat{K}_j)^2 \Psi_j(\mathbf{r}) = 0 \tag{5}$$

其中 $j = 1, 2, \ldots, n+1$，且 $\hat{K}_j$ 与 $z$ 无关：

$$\hat{K}_j(\mathbf{R}) = K_0 \sqrt{1 + \frac{\Delta}{(2\pi K_0)^2} + \frac{\sigma}{\pi K_0} U_j(\mathbf{R})} \tag{6}$$

因此切片 $j$ 内的波函数（从 $z = (j-1)\varepsilon$ 到 $z = j\varepsilon$）为：

$$\Psi_j(\mathbf{R}, z) = e^{2\pi i \hat{K}_j z'}\Phi_j^f[\mathbf{R}, (j-1)\varepsilon] + e^{-2\pi i \hat{K}_j z'}\Phi_j^b[\mathbf{R}, (j-1)\varepsilon] \tag{7}$$

其中 $z' = z - (j-1)\varepsilon$，$\Phi_j^f$ 和 $\Phi_j^b$ 分别表示 $z = (j-1)\varepsilon$ 处的前向散射波和背散射波。**需要注意**：由于 $\hat{K}_j$ 是作用于 $(x, y)$ 的算子，方程 (7) 是方程 (5) 的正确一般解的唯一形式。

### 2.3 边界条件与传递矩阵

对于整个问题，已知的是固体势函数和边界条件：
1. **入射面**: $\Phi_1^f(\mathbf{R}, z=0) = e^{2\pi i \mathbf{K}_{0\parallel} \cdot \mathbf{R}}$，其中 $\mathbf{K}_{0\parallel}$ 表示倾斜照明（垂直入射时 $\mathbf{K}_{0\parallel} = 0$，$\Phi_1^f = 1$）
2. **出射面**: $\Phi_{n+1}^b(\mathbf{R}, z=n\varepsilon) = 0$

最终需要知道的是 $z = n\varepsilon$ 处的前向散射波和 $z = 0$ 处的背散射波，即 $\Phi_{n+1}^f(\mathbf{R}, z=n\varepsilon)$ 和 $\Phi_1^b(\mathbf{R}, z=0)$。

由于波函数在相邻两切片的界面处应平滑连接，在 $z = (j-1)\varepsilon$ 处有：

$$\Psi_{j-1}|_{z=(j-1)\varepsilon} = \Psi_j|_{z=(j-1)\varepsilon} \tag{8}$$

$$\frac{\partial \Psi_{j-1}}{\partial z}\bigg|_{z=(j-1)\varepsilon} = \frac{\partial \Psi_j}{\partial z}\bigg|_{z=(j-1)\varepsilon} \tag{9}$$

**切片传输算子（STO）矩阵**定义为：

$$\mathscr{S}_{j,j-1} = \begin{pmatrix} \mathscr{S}_{j,j-1}^{11} & \mathscr{S}_{j,j-1}^{12} \\ \mathscr{S}_{j,j-1}^{21} & \mathscr{S}_{j,j-1}^{22} \end{pmatrix} \tag{12}$$

其中**前向散射系数（FSC）算子** $\mathscr{F}_{j,j-1}$ 和**背散射系数（BSC）算子** $\mathscr{B}_{j,j-1}$：

$$\mathscr{F}_{j,j-1} = \frac{\hat{K}_j + \hat{K}_{j-1}}{2\hat{K}_j}, \quad \mathscr{B}_{j,j-1} = \frac{\hat{K}_j - \hat{K}_{j-1}}{2\hat{K}_j} \tag{13}$$

利用传递矩阵技术，得到多层传输算子（MSTO）矩阵 $\mathscr{T}$：

$$\mathscr{T} = \prod_{j=2}^{n+1} \mathscr{S}_{j,j-1} \tag{15}$$

从边界条件 $\Phi_{n+1}^b = 0$，最终得到严格解：

$$\Phi_1^b = -(\mathscr{T}^{22})^{-1} \mathscr{T}^{21} \Phi_1^f \tag{19}$$

$$\Phi_{n+1}^f = \mathscr{T}^{11} - \mathscr{T}^{12}(\mathscr{T}^{22})^{-1} \mathscr{T}^{21}\Phi_1^f \tag{20}$$

至此，完整薛定谔方程在多层方案中得以严格求解。一旦知道固体薄膜的势函数和入射波函数，上表面的背散射波和下表面的透射波原则上可从方程 (19) 和 (20) 计算得出。

---

## 3. 对传统前向散射近似的修正

### 3.1 前向散射薛定谔方程的约化

假设背散射效应可忽略，即令 FSC 算子 $\mathscr{F}_{j,j-1} = 1$ 和 BSC 算子 $\mathscr{B}_{j,j-1} = 0$，得到 $\Phi_j^b = 0$ 且：

$$\Phi_{n+1}^f(\mathbf{R}) = \left[\prod_{j=1}^{n} e^{2\pi i \hat{K}_j(\mathbf{R})\varepsilon}\right] \Phi_1^f(\mathbf{R}) \tag{21}$$

当切片厚度趋近于零时，这是以下**约化薛定谔方程**的精确解：

$$\frac{\partial \Phi(\mathbf{R}, z)}{\partial z} = 2\pi i \hat{K}(\mathbf{R}, z) \Phi(\mathbf{R}, z) = 2\pi i K_0 \left[\sqrt{1 + \frac{\Delta}{(2\pi K_0)^2} + \frac{\sigma}{\pi K_0} U(\mathbf{R}, z)}\right] \Phi(\mathbf{R}, z) \tag{22}$$

令 $\Phi(\mathbf{R}, z) = \varphi(\mathbf{R}, z)e^{2\pi i K_0 z}$ 并代入方程 (22)，展开平方根算子得到：

$$\frac{\partial \varphi}{\partial z} = i\left\{2\pi K_0\left[\sqrt{1 + \frac{\Delta}{(2\pi K_0)^2}} - 1\right] + \sigma U(\mathbf{R}, z) - \frac{\sigma}{(4\pi K_0)^2}(\Delta U + U\Delta) + \frac{\sigma}{4\pi K_0} \nabla U \cdot \nabla + \cdots\right\}\varphi \tag{23}$$

因此可以看到：
1. 忽略背散射效应后，薛定谔方程可约化为 $z$ 的一阶微分方程
2. 传统的前向散射高能近似仅是方程 (23) 在电子波长上的**一阶近似**：

$$\frac{\partial \varphi}{\partial z} = i\left\{\frac{\Delta}{4\pi K_0} + \sigma U(\mathbf{R}, z)\right\}\varphi(\mathbf{R}, z) \tag{24}$$

已注意到，使用抛物面近似来计算激发误差，方程 (24) 可能在 HOLZ 图案计算中引起可观察的误差。因此，在基于方程 (24) 的实空间 Bloch 波（微扰）方法和 CMS 中，应使用修正形式的"抛物面"激发误差。这一修正可以方便地包含在方程 (24) 中：

$$\frac{\partial \varphi}{\partial z} = i\left\{2\pi K_0\left[\sqrt{1 + \frac{\Delta}{(2\pi K_0)^2}} - 1\right] + \sigma U(\mathbf{R}, z)\right\}\varphi(\mathbf{R}, z) \tag{25}$$

**图 2**: HEED 中抛物面近似的说明 — 抛物面与 Ewald 球的示意图。

然而，方程 (23) 表明，这样的修正仍然不够精确。展开方程 (23) 的第二项：

$$\sqrt{1 + \frac{\Delta}{(2\pi K_0)^2} + \frac{\sigma}{\pi K_0}U} \approx 1 + \frac{\Delta}{2(2\pi K_0)^2} + \frac{\sigma}{2\pi K_0}U - \frac{\Delta^2}{8(2\pi K_0)^4} - \frac{\sigma}{8\pi K_0(2\pi K_0)^2}(\Delta U + U\Delta - 2\nabla U \cdot \nabla) + \cdots \tag{27}$$

其中 $\nabla$ 是 $x$-$y$ 平面的梯度算子。因此，如果 $\Delta^2/(4\pi K_0)^3$ 项是从抛物面到球面的重要修正，那么混合项 $\sigma/(4\pi K_0)^3[\Delta U + (\nabla U) \cdot \nabla]$ 在激发误差修正中也可能同样重要，因为两者都正比于 $\lambda^3$。理论上，忽略这些混合项将导致 HOLZ 图案精细结构的计算误差。换句话说，**Ewald 球实际上被晶体势函数轻微调制**。

### 3.2 对 CMS 的修正以计算 HOLZ 图案

为了在 CMS 中包含更精确的修正，从方程 (21) 出发，对切片传输算子做以下高能近似和多层近似：

$$e^{2\pi i(\hat{K}_j - K_0)\varepsilon} \approx e^{i\varepsilon\left[2\pi K_0\left(\sqrt{1+\Delta/(2\pi K_0)^2}-1\right) + \sigma U_j - \frac{\sigma}{(4\pi K_0)^2}(\Delta U_j + U_j\Delta) + \mathcal{O}\left(U_j - \frac{\sigma U_j^2}{4\pi K_0}\right)\right]} \tag{28}$$

其中**纯传播子** $p(\mathbf{R})$ 和**纯相位光栅** $q_j(\mathbf{R})$ 分别为：

$$p(\mathbf{R}) = e^{2\pi i \varepsilon K_0\left[\sqrt{1 + \Delta/(2\pi K_0)^2} - 1\right]} \tag{29}$$

$$q_j(\mathbf{R}) = e^{i\varepsilon\sigma\left[U_j - (\sigma U_j^2)/(4\pi K_0)\right]} \tag{30}$$

而**混合算子** $o_j(\mathbf{R})$ 为：

$$o_j(\mathbf{R}) = e^{-i\varepsilon\frac{\sigma}{(4\pi K_0)^2}(\Delta U_j + U_j\Delta)} \tag{31}$$

在倒空间中，传播子取以下形式：

$$P(\mathbf{K}) = e^{2\pi i \varepsilon K_0\left[\sqrt{1 + \mathbf{K}^2/K_0^2} - 1\right]} \tag{32}$$

因此得到**修正的多层公式**：

$$\Phi_{n+1}^f(\mathbf{R}) = \left[\prod_{j=1}^{n} p(\mathbf{R}) o_j(\mathbf{R}) q_j(\mathbf{R})\right] \Phi_1^f(\mathbf{R}) \tag{33}$$

可以看到，没有混合算子时（即 $o_j(\mathbf{R}) = 1$），方程 (33) 就退化为 CMS，可使用 FFT 技术方便地计算。

对于混合算子的计算，有两种情况：
1. **计算大角度共振 HOLZ 反射**：切片厚度需取得足够小，使得可以使用近似 $o_j(\mathbf{R}) \approx 1 - i\varepsilon\frac{\sigma}{(4\pi K_0)^2}(\Delta U_j + U_j\Delta)$，每层需要比 CMS 多两次 FFT
2. **计算 ZOLZ 反射和低阶 HOLZ 效应**：可忽略混合算子，方程 (33) 退化为 CMS，可使用较大切片厚度（如果包含原子 Debye-Waller 因子）

---

## 4. 包含背散射效应的多层方法

### 4.1 计算背散射的高能近似

虽然第 2 节给出的类薛定谔方程的严格解在原则上适用于从 LEED 到 HEED 的固体中所有情形（忽略自旋效应），但随着入射电子能量的降低，显式计算变得越来越困难。然而，对于高能电子（如 $\geq 100$ keV），如第 3 节所述，波矢算子展开到二阶（正比于 $\lambda^3$）对大多数 HEED 情形已足够精确：

$$\hat{K}_j \approx K_0\left[1 + \frac{\Delta}{2(2\pi K_0)^2} + \frac{\sigma}{2\pi K_0}U_j - \frac{\Delta^2}{8(2\pi K_0)^4} - \frac{\sigma}{8\pi K_0(2\pi K_0)^2}(\Delta U_j + U_j\Delta) + \frac{1}{2}\left(\frac{\sigma U_j}{2\pi K_0}\right)^2\right] \tag{36}$$

实际上，如果不需要对激发误差进行精确修正，甚至传统高能近似也是足够的：

$$\hat{K}_j \approx K_0\left[1 + \frac{\Delta}{2(2\pi K_0)^2} + \frac{\sigma}{2\pi K_0}U_j\right] \tag{37}$$

类似地，对于 HEED，BSC 算子可近似为：

$$\mathscr{B}_{j,j-1} = \frac{\hat{K}_j - \hat{K}_{j-1}}{2\hat{K}_j} \approx \frac{\sigma}{4\pi K_0}(U_j - U_{j-1}) \tag{38}$$

可以看到，**背散射波的振幅与电子波长的平方成正比**（$\mathscr{B}_{j,j-1} \propto \lambda^2$），而前向散射波的振幅（对于 ZOLZ 反射）与电子波长成正比（见相位光栅的展开）。因此对于 HEED，背散射效应一般远弱于前向散射效应。

这提示我们使用所谓的**"单次背散射近似"**：对于 STO 矩阵，可设 $\mathscr{S}_{j,j-1}^{11} = 0$ 且 $\mathscr{S}_{j,j-1}^{22} = e^{-2\pi i \hat{K}_{j-1}\varepsilon}$。在此近似下，基本思想是双重（或多重）背散射效应可忽略（正比于 $\lambda^4$）。

简化后的 STO 矩阵为：

$$\mathscr{S}_{j,j-1} \approx \begin{pmatrix} (1 - \mathscr{B}_{j,j-1})e^{2\pi i \hat{K}_{j-1}\varepsilon} & \mathscr{B}_{j,j-1} e^{2\pi i \hat{K}_{j-1}\varepsilon} \\ \mathscr{B}_{j,j-1} e^{-2\pi i \hat{K}_{j-1}\varepsilon} & e^{-2\pi i \hat{K}_{j-1}\varepsilon} \end{pmatrix} \tag{39}$$

### 4.2 计算背散射效应的多层公式

为找到计算背散射波的多层公式，将方程 (39) 代入 MSTO 矩阵。经过数学推导，最终得到：

**前向散射波**（方程 47）:

$$\Phi_{n+1}^f = \left[\prod_{j=2}^{n+1} (1 - \mathscr{B}_{j,j-1}) e^{2\pi i \hat{K}_{j-1}\varepsilon}\right] \Phi_1^f \tag{47}$$

**背散射波**（方程 48）:

$$\Phi_1^b = \sum_{j=1}^{n} \left[\prod_{k=j+1}^{n+1} e^{2\pi i \hat{K}_{k-1}\varepsilon}\right] \mathscr{B}_{j+1,j} \left[\prod_{l=2}^{j} (1 - \mathscr{B}_{l,l-1}) e^{2\pi i \hat{K}_{l-1}\varepsilon}\right] \Phi_1^f \tag{48}$$

**图 3**: 方程 (47) 前向散射和方程 (48) 背散射的示意图。

整个过程的物理含义如下（图 3）：
- **前向散射波**: 波函数逐层传播，每层首先被 $(1 - \mathscr{B}_{j,j-1})$ 因子修正（即扣除背散射损失的部分），然后向前传播 $e^{2\pi i \hat{K}_{j-1}\varepsilon}$
- **背散射波**: 是每个切片界面处产生的背散射贡献（$\mathscr{B}_{j+1,j}$）通过所有上覆切片反向传播到上表面的累积和

对于 HEED，波矢算子和 BSC 算子可分别用方程 (36) 和 (38) 近似，因此已为 CMS 建立的技术仍可用于计算。

---

## 5. 讨论

### 5.1 多层理论

CMS 在计算电子显微学中一直是一个强大的方法，但其在某些 HEED 情形（如 HOLZ 效应和大角度倾斜照明）中的适用性从未令人信服。其根本原因在于它基于约化薛定谔方程（方程 24）。然而，我们已看到多层理论可从弹性电子散射的完整薛定谔方程出发来建立。这一发展使得理论在必要时可以包含更精确的效应。因此，多层理论在所有弹性电子散射情形中原则上可以达到与 Bloch 波理论（但该理论不能包含表面效应）和 Green 函数理论同等的精度。

有趣的是，在 RHEED 计算中存在另一种多层法，即所谓的 **R 矩阵多层理论**。原则上，R 矩阵多层理论和本文提出的算子多层理论应该是等价的，因为两者都基于完整薛定谔方程。然而，算子多层理论是 Cowley-Moodie 多层理论的进一步发展，其中相位光栅和传播子的物理光学概念在数值计算中起关键作用。而 R 矩阵多层理论没有这样的概念，其本质上是一种倒空间矩阵方法。另一方面，在本文中，我们主要关注透射波，背散射效应仅基于"单次背散射近似"被包含（虽然方程 19 和 20 对所有散射效应都精确有效）。

### 5.2 前向散射近似

CBED 情形中 HOLZ 图案的精确模拟对于精确测量局域晶格参数非常重要。目前，HOLZ 图案的实空间 Bloch 波计算基于前向散射的传统高能近似，包括从抛物面到球面的修正。然而，方程 (23) 表明，如果球面修正显著，一些其他修正也应包含在内。这些修正随电子波长变化的重要性可从方程 (23) 数值检验。理论上，这些修正在以下两种情况下预期是显著的：
1. 低能或中能电子衍射
2. 在冷却晶体中——可能出现许多大角度 HOLZ 环

已表明 CMS 如何将 HOLZ 效应纳入考虑。对第 3.2 节的新方法也可做同样的分析，从而导致 ZOLZ 相互作用情形下波场的新表达式（不考虑势函数沿 z 轴的变化时）：

$$\Phi_{n+1}^f(\mathbf{R}) \approx \exp\left\{2\pi i n\varepsilon K_0\left[\sqrt{1 + \frac{\Delta}{(2\pi K_0)^2}} + \frac{\sigma}{\pi K_0}\bar{U}\right]\right\} \Phi_1^f(\mathbf{R}) \tag{50}$$

其中 $\bar{U}(\mathbf{R})$ 是整个晶体的平均势函数。然而，对于 HEED，如果固体薄膜的总厚度在几十纳米左右，方程 (50) 不会对传统的 ZOLZ 效应定义带来显著修正，因为：(i) 修正项正比于 $\lambda^3$，(ii) ZOLZ 反射位于抛物面近似已足够精确的区域（图 2）。

### 5.3 背散射与大光束倾斜

**图 4**: 晶体中电子散射的 Ewald 球构造 — 背散射与 HOLZ 效应的关系。

从 Ewald 球构造的角度看（图 4），背散射反射实际上是因沿束方向的势函数变化而产生的 HOLZ 效应。背散射效应的出现取决于：入射束方向、晶体结构、原子温度因子、晶体厚度以及电子波长。

然而，在第 4 节的公式中包含背散射效应时，首先需要澄清**背散射的概念**。在理论方案中，背散射是**相对于切片法线（z 轴）定义的**（图 5a），而不是相对于入射束方向（图 5b）。对于 HEED，相对于入射束方向定义的背散射效应很难出现，因为原子的大角度散射因子很弱。

**图 5**: 背散射效应的定义: (a) 相对于切片法线; (b) 相对于入射束方向。

对于多层理论，原则上总可将入射束方向取为切片法线，从而无需考虑束倾斜效应，也可合理忽略背散射效应。但实践中并不总是可能或方便这样做，因此束倾斜和背散射效应需要被包含在多层法中。例如：
- 计算**非正交晶体**中束沿晶带轴的透射波（图 6a）
- 计算**正交晶体**中束大幅度偏离表面法线的透射波（图 6b）

在这些情形下，如果将切片法线平行于入射束方向，则无法计算相位光栅。此时，方程 (47) 可能有用，因为它同时考虑了束倾斜和背散射效应。

**图 6**: 可能的背散射效应示意: (a) 非正交晶体电子衍射; (b) 正交晶体电子衍射; (c) RHEED 情形。

已知 CMS 对大光束倾斜不适用。从以上讨论可见，这是因为 CMS 未包含背散射效应（即使在大光束倾斜情况下）。展示背散射效应对大光束倾斜重要性的一个极端例子是 **RHEED**，其中入射束倾斜到几乎垂直于表面法线（图 6c）。但需指出，对于背散射效应占主导的 RHEED 情形，方程 (47) 和 (48) 可能不足够，因为它们基于"单次背散射近似"。

### 5.4 新多层公式的计算

为了在计算大角度共振 HOLZ 反射的多层法中包含修正算子（方程 31），需要两倍于 CMS 过程的 FFT 次数，因此总计算时间大约加倍。需要强调的是，对于用多层法计算 HOLZ 反射，切片厚度必须取得足够小，使多层势函数分布趋近于固体中的真实势函数分布，以保证计算的收敛性。

对于倾斜照明，束倾斜效应可包含在入射波函数 $\Phi_1(\mathbf{R})$ 中（通过平面波因子 $e^{2\pi i \mathbf{K}_{0\parallel} \cdot \mathbf{R}}$），或转移到多层公式中。后者便于使用 FFT 技术执行多层计算。转移方法是将所有 $\Delta\Phi(\mathbf{R})$ 操作改为 $[\Delta + 4\pi i \mathbf{K}_{0\parallel} \cdot \nabla + (2\pi i K_{0\parallel})^2]\Phi(\mathbf{R})$（实空间），或相应地将 $(2\pi i \mathbf{K})^2 \tilde{\Phi}(\mathbf{K})$ 改为 $(2\pi i)^2(\mathbf{K} + \mathbf{K}_{0\parallel})^2 \tilde{\Phi}(\mathbf{K})$（倒空间）。

从方程 (47) 可见，包含背散射效应的透射波计算可以方便地通过第 4.2 节的多层过程完成，只需将相位光栅修正为因子 $(1 - \mathscr{B}_{j,j-1}) \approx [1 - \sigma/(4\pi K_0)(U_j - U_{j-1})] \approx \exp\{-\sigma/(4\pi K_0)(U_j - U_{j-1})\}$，从而变成 $\exp\{i\varepsilon\sigma[U_j + (\sigma\Delta/(4\pi K_0\varepsilon))(U_j - U_{j-1})]\}$。然而，**背散射波本身的计算将非常耗时**，除非使用大量计算机内存保存所有切片的透射函数。幸运的是，在 HEED 的实际情形中，通常只有背散射效应对透射波的影响是感兴趣的。需要注意，背散射在透射波中表现为一种**吸收效应**，因此传统的透射束归一化检验将不精确。

---

## 6. 结论

1. 弹性电子散射的薛定谔方程可以在多层方案框架内**严格求解**。为此，切片传输算子（STO）矩阵的概念和传递矩阵技术非常有用，它们导向了一个更一般、更精确的多层理论。这一新方法**保留了传统多层法相对于 Bloch 波方法的重要计算优势**，因此对任何类型的固体结构（包括表面）都普遍有效。

2. 新方法导出了对传统高能近似的**精确修正**——传统高能近似是许多现有弹性前向散射动力学理论的基础。利用所谓的"单次背散射近似"，得到了计算透射波的**改进多层公式**。

3. 该公式**考虑了背散射效应**，因此可用于大光束倾斜。研究表明，传统多层过程的大多数数值技术，包括 **FFT 技术**，仍可用于执行新的多层过程。

4. 新公式中，背散射修正以因子 $(1 - \mathscr{B}_{j,j-1})$ 的形式自然地修正了每层的相位光栅，等价于对势函数增加了与 $\lambda^2$ 成正比的有效吸收项。

---

## 致谢

本文呈现的研究结果部分由比利时国家科学政策规划办公室发起的比利时大学间吸引力极点计划赞助。科学责任由作者承担。作者之一 JHC 感谢 J. Van Landuyt 教授和 G. Van Tendeloo 教授的持续支持和对稿件的仔细阅读。作者特别感谢李新奇博士关于传递矩阵技术的讨论。

---

## 参考文献

1. J.C.H. Spence, J.M. Zuo, *Electron Microdiffraction*, Plenum, New York, 1992.
2. Z.L. Wang, *Elastic and Inelastic Scattering in Electron Diffraction and Imaging*, Plenum, New York, 1995.
3. D. Van Dyck, Phys. Status Solidi (b) 77 (1976) 301.
4. H.A. Bethe, Ann. Phys. (Leipzig) 87 (1928) 55.
5. D.M. Bird, J. Electron Microsc. Technol. 13 (1989) 77.
6. P.B. Hirsch, A. Howie, R.B. Nicholson, D.W. Pashley, M.J. Whelan, *Electron Microscopy of Thin Crystals*, Krieger, New York, 1977.
7. C.J. Humphreys, Rep. Prog. Phys. 42 (1979) 1825.
8. H.S. Kim, S.S. Sheinin, Phys. Status Solidi (b) 109 (1982) 807.
9. L.D. Marks, Ultramicroscopy 38 (1992) 325.
10. J.M. Cowley, *Diffraction Physics*, Ch. 11, North-Holland, New York, 1981.
11. P.G. Self, M.A. O'Keefe, in: *High-Resolution Transmission Electron Microscopy and Associated Techniques*, Oxford University Press, Oxford, 1992, p. 259.
12. J.M. Cowley, A.F. Moodie, Acta Crystallogr. 10 (1957) 609.
13. K. Ishizuka, N. Uyeda, Acta Crystallogr. A 33 (1977) 740.
14. S.H. Stobbs, C.B. Boothroyd, W.M. Stobbs, Inst. Phys. Conf. Ser. No. 98, 1989, p. 387.
15. Z.L. Wang, Phys. Rev. B 41 (1990) 12818.
16. W. Coene, D. Van Dyck, Ultramicroscopy 33 (1990) 261.
17. D. Van Dyck, Adv. Elec. Elec. Phys., 1985, p. 295.
18. D.F. Lynch, A.F. Moodie, Surf. Sci. 32 (1972) 422.
19. D.F. Lynch, A.E. Smith, Phys. Status Solidi (b) 119 (1983) 355.
20. T.C. Zhao, H.C. Poon, S.Y. Tong, Phys. Rev. B 38 (1988) 1172.
21. K. Fujiwara, J. Phys. Soc. Jpn. 16 (1961) 2226.
22. K. Fujiwara, J. Phys. Soc. Jpn. 17 (Suppl. BII) (1962) 118.
23. A. Howie, J. Phys. Soc. Jpn. 17 (Suppl. BII) (1962) 122.
24. J.C.H. Spence, *Experimental High-Resolution Electron Microscopy*, Oxford University Press, New York, 1988.
25. R. Gevers, M. David, Phys. Status Solidi 2 (1982) 665.
26. J.H. Chen, Y.M. Wang, X.J. Luo, L.Y. Ding, X.L. Cheng, Phil. Mag. Lett. 71 (1995) 33.
27. J.H. Chen, M. Op De Beeck, D. Van Dyck, Microsc. Microanal. Microstruct. 7 (1996) 27.
28. Takagaki, Ferry, Phys. Rev. B 46 (1992) 15218.
29. D. Van Dyck, J. Microsc. 119 (1980) 141.
30. M.V. Berry, J. Phys. C 4 (1971) 697.
31. P.G. Self, M.A. O'Keefe, P.R. Buseck, A.E.C. Spargo, Ultramicroscopy 11 (1983) 35.
32. R. Kilaas, M.A. O'Keefe, K.M. Krishnan, Ultramicroscopy 21 (1987) 47.
33. J.R. Baker, S. McKernan, in: Electron Microscopy and Analysis 1981, Inst. Phys. Conf. Ser. 61, 1982, p. 283.
34. R. Vincent, D.M. Bird, J.W. Steeds, Phil. Mag. A 50 (1984) 765.
35. D. Van Dyck, W. Coene, Ultramicroscopy 15 (1984) 29.
36. L.C. Qin, K. Urban, Ultramicroscopy 33 (1990) 159.
37. P. Goodman, A.F. Moodie, Acta Crystallogr. A 30 (1974) 280.
38. T.C. Zhao, S.Y. Tong, Phys. Rev. B 47 (1993) 3923.
39. J.M. Zuo, Ultramicroscopy 41 (1992) 211.
40. J.W. Steeds, in: *Introduction to Analytical Electron Microscopy*, Chapter 15, Plenum, New York, 1979.
41. J.H. Chen, D. Van Dyck, M. Op de Beeck, Acta Crystallogr. A, 1997 (in press).
42. J.H. Chen, D. Van Dyck, M. Op de Beeck, J. Broeckx, J. Van Landuyt, Phys. Status Solidi (a) 150 (1995) 13.
