# Section 1: Introduction — 写作计划

> 基于 Swales CARS 模型 + 四条写作底层逻辑重构。
>
> - **漏斗结构**：每段标注漏斗宽度 `[宽] → [窄]`
> - **创造研究空间**：用 `However` / `Although` 明确标注 Gap 入口
> - **时态信号**：每段标注时态策略（现在时=公认事实，过去时=具体实验，现在完成时=对当下强相关）
> - **句际衔接**：标注段间/句间的逻辑连接策略
>
> PINN 首次出现时使用全称 Physics-Informed Neural Networks (PINNs)，之后使用缩写。

---

## Move 1: Establishing a Territory（建立领地）

### Para 1 — 漏斗入口：大背景 `[最宽]`

Elliptic interface problems, partial differential equations of the form $-\nabla\cdot(\beta\nabla u) = f$ with piecewise-constant coefficients, are fundamental to computational science and engineering. They model physical systems where material properties change sharply across an internal boundary $\Gamma$: composite materials with mismatched stiffness, immiscible fluid interfaces, and biological membranes separating distinct tissue regions \cite{peskin2002immersed, leveque1994immersed}. Without special treatment at $\Gamma$, standard numerical schemes degrade to $O(1)$ accuracy, because the coefficient jump forces a gradient discontinuity that smooth basis functions cannot capture \cite{babuvska1970finite, li2006immersed}.

### Para 2 — 经典数值方法综述 `[宽 → 中]`

Restoring accuracy requires methods that explicitly encode the jump in $\beta$. While the finite element method can achieve high-order convergence on interface-conforming meshes, generating such meshes becomes prohibitively expensive for complex or moving interfaces \cite{babuvska1970finite}. Fixed-grid approaches avoid this cost by modifying difference stencils near $\Gamma$ to enforce the jump conditions; the Immersed Interface Method (IIM) \cite{leveque1994immersed} and the Matched Interface and Boundary (MIB) method \cite{zhou2006fictitious} follow this strategy with increasing accuracy. Generalized Finite Differences (GFD) go further, abandoning structured grids altogether and reconstructing derivative operators from arbitrary node distributions via weighted least-squares \cite{liszka1980finite, benito2001influence}.

---

## Move 2: Establishing a Niche（创造研究空间）

### Para 3 — 传统方法的局限 + PINN 的兴起与局限 `[中 → 窄]`

These methods, however, share a reliance on hand-crafted stencils that must be redesigned for each new interface geometry. Physics-Informed Neural Networks (PINNs) \citep{raissi2019physics} sidestep this constraint by embedding PDE residuals directly into the loss function of a neural network, requiring neither stencils nor mesh conformity. The framework has been applied successfully to forward and inverse problems alike \citep{karniadakis2021physics, lu2021deepxde}, but interface problems expose its weaknesses. Multilayer perceptrons are biased toward low-frequency solutions \cite{rahaman2019spectral}, making it difficult to resolve the sharp gradients that arise at $\Gamma$. Multi-term losses (PDE residual, boundary conditions, interface jumps) compete for gradient magnitude, often destabilizing training \cite{wang2021understanding}; in some configurations, the network fails to converge entirely \cite{krishnapriyan2021characterizing}. Domain decomposition \cite{jagtap2020conservative} and discontinuity-aware architectures \cite{hu2022discontinuity} offer partial remedies at the cost of additional complexity. In our preliminary experiments, a related issue emerged: when we refined the grid from $N=24$ to $N=32$, the PINN's error did not decrease, suggesting that autograd-computed derivatives cannot capitalize on finer spatial discretizations.

### Para 4 — GNN 的发展与 Gap `[窄]`

Graph Neural Networks (GNNs) \cite{kipf2017semi, hamilton2017inductive} offer a natural alternative for PDE problems on unstructured domains, since their message-passing operations \cite{gilmer2017neural} directly encode the topology of the computational mesh. In physics simulation, graph-based models have learned complex dynamics from mesh-level interactions \cite{sanchez2020learning, pfaff2020learning}, and message passing has been applied directly to PDE solution operators \cite{brandstetter2022message}. Fourier Neural Operators \cite{li2020fourier} take a different route by learning in spectral space, but they require uniform grids, a constraint incompatible with interface-adapted meshes. Yet none of these approaches exploit the derivative information that classical numerical stencils can provide. GFD stencils encode precise local gradient structure; GNNs offer flexible nonlinear approximation on graphs. The possibility of combining these two capabilities, using precomputed stencil coefficients as message-passing weights, remains unexplored.

---

## Move 3: Occupying the Niche（占据研究空间）

### Para 5 — 本文贡献 `[最窄 → 漏斗底部]`

We propose G-GRN (Graph-based Generalized Regression Network) to fill this gap. G-GRN treats precomputed GFD stencil coefficients as edge weights in a graph neural network, replacing autograd with scatter-based derivative approximation. Because derivative accuracy now depends on stencil order rather than network smoothness, the method naturally benefits from mesh refinement, a property that standard PINNs lack. A resolution-normalization scheme (scaling by $h_{\text{char}} = 2/(N-1)$) keeps stencil-derived features at $O(1)$ across grid densities, preventing feature magnitudes from drifting as the mesh is refined. Training minimizes a strong-form loss combining the PDE residual, Dirichlet boundary conditions, and both interface transmission conditions ($[u]$ and $[\beta\partial_n u]$), with Fourier feature encoding \cite{tancik2020fourier} to counteract the spectral bias of the underlying MLP. Experiments on three interface geometries (circular, oscillating, elliptic) confirm that G-GRN outperforms standard PINNs in both accuracy and training speed.

### Para 6 — 论文路线图（1-2 句）

The remainder of this paper is organized as follows. Section \ref{sec:problem} formulates the elliptic interface problem. Section \ref{sec:method} details the G-GRN architecture, including graph construction, GFD stencil computation, the network design, and the training protocol. Section \ref{sec:experiments} presents numerical experiments on three test cases, and Section \ref{sec:conclusion} concludes with a discussion of limitations and future directions.

---

## 整体漏斗结构一览

```
Para 1  [████████████████████]  计算科学中的界面问题（公认事实，所有读者的集会点）
Para 2  [██████████████████]    经典数值方法如何处理（学术版图）
Para 3  [████████████████]      However 传统局限 → PINN 兴起 → However PINN 局限（双重 Gap）
Para 4  [██████████████]        GNN 发展 → However 未利用 GFD（第三个 Gap → 交叉点空白）
Para 5  [████████████]          In this work, we propose G-GRN...（占据空间，贡献列表）
Para 6  [██████████]            论文路线图
```

## 句际衔接关键词索引

| 位置 | 衔接词 | 功能 |
|---|---|---|
| Para 1 → 2 | "To address this challenge, ..." | 从困难引出解决方案 |
| Para 2 内部 | "Alternatively, ..." / "More recently, ..." | 方法间并列 |
| Para 2 → 3 | "**However**, these approaches share ..." | Gap 1 入口 |
| Para 3 内部 | "To overcome ..., PINNs ..." | 从局限引出 PINN |
| Para 3 内部 | "**Despite their success**, PINNs face ..." | Gap 2 入口 |
| Para 3 → 4 | "**An alternative paradigm** ..." | 引出另一路径 |
| Para 4 末尾 | "**Yet**, existing GNN-based solvers ..." | Gap 3 入口 |
| Para 4 → 5 | "**To bridge this gap**, we propose ..." | 占据研究空间 |

---

## 可用文献索引（reference.bib）

| 主题 | Citation Key | 用途 |
|---|---|---|
| PINN 开山 | `raissi2019physics` | Para 3：首次引用 PINN（全称） |
| PINN 综述 | `karniadakis2021physics` | Para 3：PINN 广泛应用 |
| DeepXDE | `lu2021deepxde` | Para 3：PINN 库/应用 |
| PINN 梯度病态 | `wang2021understanding` | Para 3：训练困难 |
| PINN 失败模式 | `krishnapriyan2021characterizing` | Para 3：收敛失败 |
| 界面 PINN | `jagtap2020conservative` | Para 3：cPINN 分域处理 |
| 界面 NN | `hu2022discontinuity` | Para 3：浅层网络处理不连续 |
| 谱偏差 | `rahaman2019spectral` | Para 3：MLP 低频偏好 |
| Fourier features | `tancik2020fourier` | Para 5：缓解谱偏差 |
| GNN 原始 | `scarselli2008graph` | Para 4：GNN 基础 |
| GCN | `kipf2017semi` | Para 4：图卷积网络 |
| GraphSAGE | `hamilton2017inductive` | Para 4：归纳式图学习 |
| MPNN | `gilmer2017neural` | Para 4：消息传递框架 |
| 关系归纳偏置 | `battaglia2018relational` | Para 4：GNN 统一框架 |
| Learning to Simulate | `sanchez2020learning` | Para 4：GNN 物理模拟 |
| MeshGraphNets | `pfaff2020learning` | Para 4：GNN 网格模拟 |
| MP-PDE Solvers | `brandstetter2022message` | Para 4：GNN 直接求解 PDE |
| FNO | `li2020fourier` | Para 4：Neural Operator |
| IBM | `peskin2002immersed` | Para 1：经典界面方法 |
| IIM | `leveque1994immersed` | Para 1-2：经典界面方法 |
| MIB | `zhou2006fictitious` | Para 2：MIB 方法 |
| GFD | `liszka1980finite` | Para 1-2：GFD 基础 |
| GFD 分析 | `benito2001influence` | Para 2：GFD 影响因素 |
| FEM 界面 | `babuvska1970finite` | Para 1-2：FEM 局限 |
