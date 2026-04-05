# 3 Methodology

The G-GRN framework consists of three stages: constructing a topology-aware graph from the computational domain, precomputing GFD stencils on this graph, and training a neural operator that minimizes a physics-informed residual loss.

## 3.1 Graph Construction

### Para 1 — 节点生成

We represent the computational domain as a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where the node set $\mathcal{V} = \{\mathbf{p}_i\}_{i=1}^{N}$ combines a uniform background grid with targeted refinement along the interface. First, a Cartesian grid of resolution $n \times n$ is generated to cover the computational domain $\Omega = [-1,1]^2$. Second, to resolve the sharp transition near the discontinuity, we add concentric rings of nodes on both sides of the interface. For each layer $l = 1, \ldots, L_r$, we place $N_\Gamma$ nodes uniformly in the angular direction at radii $R \pm \delta_l$, where the offset is
\begin{equation}
    \delta_l = \frac{w}{2} \cdot \frac{l}{L_r},
\end{equation}
and $w$ is the width of the refinement band. In our default setting, $L_r = 2$, $N_\Gamma = 64$, and $w = 0.1$, producing four rings of 64 nodes within a band of width $0.1$ centered on $\Gamma$. The final node set is the union of the grid nodes and the refinement nodes. Each node $i$ carries a feature vector $\mathbf{v}_i = [x_i, y_i, \phi_i, z_i]^\top \in \mathbb{R}^4$, where $\phi_i$ is the level-set value and $z_i = \text{sgn}(\phi_i) \in \{-1, 1\}$ is the region indicator.

### Para 2 — 边连接

Once the nodes are placed, we establish connectivity via a spatial radius search. An edge exists between node $i$ and node $j$ if their Euclidean distance is less than a cutoff radius $r_c$, excluding self-loops. This radius must be large enough to provide at least 5 neighbors per node for quadratic GFD fitting. We set $r_c = 4.5 / n$, where $n$ is the grid resolution. Since the background spacing is $h = 2/n$ on the domain $[-1,1]^2$, this gives $r_c \approx 2.25 h$, which typically yields 15--25 neighbors for interior nodes.

### Para 3 — 边分类

Treating all edges uniformly is insufficient for interface problems, so we partition $\mathcal{E}$ into two disjoint subsets based on the region indicator. \textit{Homophilous edges} connect nodes within the same subdomain ($z_i = z_j$); they represent connectivity within a continuous medium and are used exclusively for GFD stencil aggregation. \textit{Heterophilous edges} connect nodes across the interface ($z_i \neq z_j$); because differentiation across the discontinuity is ill-posed, these edges instead carry the interface jump constraints. This partition is encoded in a binary edge attribute $e_{ij} \in \{0, 1\}$ that guides the message-passing mechanism. The combination of dense near-interface sampling and explicit edge classification provides the spatial resolution needed to evaluate flux jumps accurately where the solution is least smooth.

## 3.2 Generalized Finite Difference Stencils

### Para 4 — WLS 推导

Evaluating the PDE residual requires accurate spatial derivatives on an unstructured node set. We use the Generalized Finite Difference (GFD) method \cite{liszka1980finite, benito2001influence}, which reconstructs differential operators from arbitrary node distributions by solving a local Weighted Least Squares (WLS) problem. Consider an interior node $i$ at position $\mathbf{p}_i$ and its homophilous neighbors $\mathcal{N}(i)$. Assuming the solution $u$ is locally smooth ($u \in C^2$), the second-order Taylor expansion for a neighbor $j$ is:
\begin{equation}
    u_j \approx u_i + \Delta x_j u_x|_i + \Delta y_j u_y|_i + \frac{\Delta x_j^2}{2} u_{xx}|_i + \Delta x_j \Delta y_j u_{xy}|_i + \frac{\Delta y_j^2}{2} u_{yy}|_i,
\end{equation}
where $\Delta x_j = x_j - x_i$ and $\Delta y_j = y_j - y_i$. To solve for the derivatives, we construct a linear system that minimizes the weighted approximation error. Let $\mathbf{q}_j = [\Delta x_j, \Delta y_j, \frac{1}{2}\Delta x_j^2, \Delta x_j \Delta y_j, \frac{1}{2}\Delta y_j^2]^\top$ be the geometric basis vector. We define the system matrix $\mathbf{V}$, the weight matrix $\mathbf{W}$, and the unknown derivative vector $\mathbf{d}$ as:
\begin{equation}
    \mathbf{V} = [\mathbf{q}_1, \dots, \mathbf{q}_k]^\top, \quad
    \mathbf{W} = \text{diag}(w_1, \dots, w_k), \quad
    \mathbf{d} = [u_x, u_y, u_{xx}, u_{xy}, u_{yy}]^\top,
\end{equation} 
where $\boldsymbol{\delta} = [\delta u_1, \dots, \delta u_k]^\top$ represents the finite differences. The weights $w_j = \phi(\|\mathbf{p}_j - \mathbf{p}_i\|)$ are determined by a decaying kernel function $\phi$ (e.g., Gaussian) to prioritize closer neighbors. The optimal derivatives are obtained by solving the normal equations:
\begin{equation}
    \mathbf{d} = (\mathbf{V}^\top \mathbf{W}^2 \mathbf{V})^{-1} \mathbf{V}^\top \mathbf{W}^2 \boldsymbol{\delta}.
\end{equation}
This yields the derivatives as weighted sums of neighbor differences. For instance, the discrete Laplacian weights $w_{ij}^{\Delta}$ are derived from the rows corresponding to $u_{xx}$ and $u_{yy}$ in the pseudo-inverse matrix $\mathbf{V}^+_{\mathbf{W}} = (\mathbf{V}^\top \mathbf{W}^2 \mathbf{V})^{-1} \mathbf{V}^\top \mathbf{W}^2$.

### Para 5 — 一致性命题

We theoretically characterize the approximation accuracy of this scheme.
\begin{proposition}[Consistency of GFD Stencil]
\label{prop:consistency}
Let $h = \max_{j \in \mathcal{N}(i)} \|\mathbf{p}_j - \mathbf{p}_i\|$ be the local characteristic length. Assuming the node distribution ensures that $\mathbf{V}^\top \mathbf{W}^2 \mathbf{V}$ is invertible, the GFD approximation is consistent. The local truncation error for second-order derivatives scales as $\mathcal{O}(h)$ for general irregular stencils and improves to $\mathcal{O}(h^2)$ for symmetric neighbor distributions.
\end{proposition}

### Para 6 — 自适应阶数与实现

However, irregular node distributions can lead to numerical instability. We employ an \textit{adaptive order strategy} based on the condition number $\kappa(\mathbf{V}) = \sigma_{\max}(\mathbf{V}) / \sigma_{\min}(\mathbf{V})$, where $\sigma$ denotes the singular values. If $\kappa(\mathbf{V})$ exceeds a stability threshold (e.g., $10^3$) or if the node has insufficient neighbors ($k < 5$), we automatically fall back to a first-order approximation by truncating the basis to linear terms only. This ensures robust gradient estimation even in degenerate geometric configurations.

### Para 7 — 同相限制与稀疏实现

Stencils are restricted to homophilous edges ($z_i = z_j$) to satisfy the $C^2$ continuity assumption of the Taylor expansion; physics across the interface is handled separately by the jump conditions. The stencil coefficients are precomputed once before training. At each training step, the PDE residual reduces to a sparse matrix-vector product on the GPU:
\begin{equation}
    \mathcal{R}(\mathbf{u}^{(t)}) = \mathbf{L} \mathbf{u}^{(t)} - \mathbf{f},
\end{equation}
where $\mathbf{L} \in \mathbb{R}^{N \times N}$ is the sparse Laplacian matrix assembled from $w_{ij}^{\Delta}$. This operation maps directly to optimized sparse kernels (e.g., \texttt{torch.sparse.mm}).

## 3.3 Network Architecture

### Para 8 — 架构总览

The G-GRN architecture approximates the solution operator by embedding the discretized differential structure into the neural message passing. Figure~\ref{fig:architecture} illustrates the complete training pipeline: from graph construction and feature assignment, through the G-GRN processor layers, to the physics-informed loss evaluation.

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{images/model-struc.pdf}
    \caption{Complete training pipeline for the G-GRN method solving elliptic interface problems $-\nabla \cdot (\beta \nabla u) = f$ with discontinuous $\beta$ across interface $\Gamma$. The graph is constructed with radius based connectivity (homo / hetero edges). Node features are projected through a linear layer and decoded via an MLP. Five physics-informed loss terms enforce PDE residuals, boundary conditions, interface jump conditions (via GFD stencils), and optional supervised data.}
    \label{fig:architecture}
\end{figure}

### Para 9 — 输入与编码

The network takes as input the node feature $\mathbf{v}_i \in \mathbb{R}^4$ defined in Section~\ref{sec:graph_construction}. The first G-GRN layer maps $\mathbf{v}_i$ from $\mathbb{R}^4$ to the hidden dimension $\mathbb{R}^H$ without a separate encoding stage; the derivative aggregation within each layer (described below) supplies gradient and curvature information, reducing the need for explicit frequency encoding at the input.

### Para 10 — G-GRN 层

The core computation takes place in $L$ stacked G-GRN layers. Unlike standard graph convolution layers that learn abstract edge weights, our layer explicitly aggregates features to approximate differential operators. In layer $\ell$, for each node $i$, we compute the gradient and Laplacian of the current feature $\mathbf{h}_i^{(\ell)}$ using the frozen GFD stencils over homophilous neighbors:
\begin{equation}
    \mathcal{D}[\mathbf{h}^{(\ell)}]_i = \sum_{j \in \mathcal{N}_{homo}(i)} w_{ij}^{\mathcal{D}} (\mathbf{h}_j^{(\ell)} - \mathbf{h}_i^{(\ell)}), \quad \text{for } \mathcal{D} \in \{ \partial_x, \partial_y, \Delta \}.
\end{equation}
These derivative features are concatenated with the node's state to form a physics-informed context vector $\mathbf{c}_i = [\mathbf{h}_i^{(\ell)}, \nabla \mathbf{h}_i^{(\ell)}, \Delta \mathbf{h}_i^{(\ell)}] \in \mathbb{R}^{4C}$, where $C$ denotes the hidden dimension. This vector is processed by a three-layer MLP that progressively reduces the dimension ($4C \to 2C \to C \to C$), with Layer Normalization and GELU activation after each of the first two linear layers, followed by a residual connection:
\begin{align}
    \mathbf{z}_1 &= \text{GELU}\bigl(\text{LN}(\mathbf{W}_1 \mathbf{c}_i)\bigr), \notag \\
    \mathbf{z}_2 &= \text{GELU}\bigl(\text{LN}(\mathbf{W}_2 \mathbf{z}_1)\bigr), \\
    \mathbf{h}_i^{(\ell+1)} &= \mathbf{h}_i^{(\ell)} + \mathbf{W}_3 \mathbf{z}_2, \notag
\end{align}
where $\mathbf{W}_1 \in \mathbb{R}^{2C \times 4C}$, $\mathbf{W}_2 \in \mathbb{R}^{C \times 2C}$, and $\mathbf{W}_3 \in \mathbb{R}^{C \times C}$.
This design mirrors a learnable numerical scheme: the network observes the local Taylor expansion (via derivatives) and learns a nonlinear correction to evolve the solution.

### Para 11 — Decoder 与设计哲学

After $L$ layers, a shallow MLP decoder maps the final embeddings to the scalar prediction $\hat{u}$. The derivative aggregation is parallelized through the \texttt{scatter\_add} primitive: treating $w_{ij}^{\mathcal{D}}(\mathbf{h}_j - \mathbf{h}_i)$ as a directional message, the stencil application reduces to a sparse matrix-matrix multiplication that scales linearly with the number of edges. This design enforces a separation of concerns: the fixed GFD stencils encode the \textit{geometry} (irregular mesh, stencil weights), while the learnable parameters capture the \textit{physics} (solution structure), freeing the network from learning basic discretization rules.

## 3.4 Physics-Informed Loss Formulation

### Para 12 — 总损失

Training is driven by a composite loss function that enforces the governing PDE, boundary conditions, interface constraints, and optional data supervision. Let $\hat{u}_\theta$ denote the network prediction. We optimize the parameters $\theta$ by minimizing a total loss $\mathcal{L}$:
\begin{equation}
    \mathcal{L}(\theta) = \lambda_{\text{pde}} \mathcal{L}_{\text{pde}} + \lambda_{\text{bc}} \mathcal{L}_{\text{bc}} + \lambda_{\Gamma} (\mathcal{L}_{J_1} + \mathcal{L}_{J_2}) + \lambda_{\text{data}} \mathcal{L}_{\text{data}},
\end{equation}
where $\lambda_{\text{pde}}, \lambda_{\text{bc}}, \lambda_{\Gamma}, \lambda_{\text{data}}$ are hyperparameters balancing the individual objectives. The first three groups encode the governing physics; the last term provides direct supervision from reference solutions.

### Para 13 — PDE 残差损失

The first component addresses the governing equation. Standard PINN formulations \cite{raissi2019physics} typically minimize the residual $r = \| -\nabla \cdot (\beta \nabla \hat{u}) - f \|^2$. However, in high-contrast media where the ratio $\beta^+ / \beta^-$ is large (e.g., $10^2$ to $10^3$), this standard form leads to \textit{gradient pathology} \cite{wang2021understanding}. The loss landscape becomes dominated by the high-$\beta$ subdomain, causing the optimizer to neglect the low-$\beta$ region. To mitigate this stiffness, we employ a normalized strong form loss. By dividing the governing equation by $\beta(\mathbf{x})$, we minimize:
\begin{equation}
    \mathcal{L}_{\text{pde}} = \frac{1}{|\mathcal{V}_{\text{int}}|} \sum_{i \in \mathcal{V}_{\text{int}}} \left\| -\Delta \hat{u}_i - \frac{f_i}{\beta_i} \right\|^2.
\end{equation}
Here, the Laplacian $\Delta \hat{u}_i$ is computed via the GFD stencils derived in Section~\ref{sec:gfd}. This normalization acts as a preconditioner, balancing the gradient magnitudes from different material phases to accelerate convergence.

### Para 14 — 跳跃损失

The transmission conditions are enforced explicitly on heterophilous edges $\mathcal{E}_{\text{hetero}}$ that cross the interface, rather than through continuous approximations. Since graph edges are directed and may traverse the interface from $\Omega^-$ to $\Omega^+$ or vice versa, we rely on the destination region indicator $z_j \in \{-1, +1\}$ to ensure consistent orientation. The discrete jump losses for value ($J_1$) and flux ($J_2$) are formulated as:
\begin{align}
    \mathcal{L}_{J_1} &= \frac{1}{|\mathcal{E}_{\text{hetero}}|} \sum_{(i,j) \in \mathcal{E}_{\text{hetero}}} \| z_j (\hat{u}_j - \hat{u}_i) - g_{1,ij} \|^2, \\
    \mathcal{L}_{J_2} &= \frac{1}{|\mathcal{E}_{\text{hetero}}|} \sum_{(i,j) \in \mathcal{E}_{\text{hetero}}} \| z_j (\beta_j \nabla \hat{u}_j \cdot \mathbf{n}_{ij} - \beta_i \nabla \hat{u}_i \cdot \mathbf{n}_{ij}) - g_{2,ij} \|^2.
\end{align}
The factor $z_j$ guarantees that the difference term always represents the canonical jump $[u] = u^+ - u^-$. To ensure high-order geometric accuracy, the interface normal $\mathbf{n}_{ij}$ is evaluated at the edge midpoint $\mathbf{x}_{mid} = (\mathbf{x}_i + \mathbf{x}_j)/2$ using the level-set gradient.

### Para 15 — BC、数据损失与默认权重

Dirichlet boundary conditions are imposed softly via $\mathcal{L}_{\text{bc}} = \frac{1}{|\mathcal{V}_{\text{bd}}|} \sum \| \hat{u}_i - g_D \|^2$. We also include a supervised data loss
\begin{equation}
    \mathcal{L}_{\text{data}} = \frac{1}{N} \sum_{i=1}^{N} \| \hat{u}_i - u_i^* \|^2,
\end{equation}
where $u_i^*$ is the reference value at node $i$. This term provides a direct regression target that is particularly useful near the interface, where the GFD stencil approximation error is largest. In our default configuration, we set $\lambda_{\text{pde}} = 1$, $\lambda_{\text{bc}} = 200$, $\lambda_{\Gamma} = 10$, and $\lambda_{\text{data}} = 1000$. The implementation also allows independent weights for the value jump ($\lambda_{J_1}$) and flux jump ($\lambda_{J_2}$); when not specified, both default to $\lambda_{\Gamma}$. For extreme contrast ratios ($\beta^+/\beta^- > 100$), $\lambda_{\text{data}}$ can be annealed during training via curriculum learning (Section~\ref{sec:training}).

## 3.5 Training Protocols \& Optimization Strategy

### Para 16 — L-BFGS 与学习率调度

Training physics-informed models requires balancing multiple competing loss terms. We optimize the network parameters using L-BFGS, a quasi-Newton method that approximates inverse Hessian information from recent gradient history. We set the initial learning rate to $\eta_0 = 1.0$, the history size to $50$, and use the strong Wolfe line search condition. This optimizer suits our setting well: the loss is evaluated over a single fixed graph without mini-batching, and the curvature information helps navigate the multi-objective loss landscape. The learning rate follows a cosine annealing schedule \cite{loshchilov2016sgdr}:
\begin{equation}
    \eta(t) = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min}) \left( 1 + \cos\frac{\pi t}{T} \right),
\end{equation}
where $t$ is the current epoch, $T$ is the total training budget (typically $5 \times 10^3$), and $\eta_{\min} = 10^{-6}$. During training, we track the model state with the lowest error. If the loss diverges to NaN, the best state is restored and training stops early.

### Para 17 — Curriculum Learning

In high-contrast scenarios ($\beta^+ / \beta^- \ge 100$) or inverted configurations, pure physics-based optimization often stagnates due to vanishing gradients in the low-diffusivity region. A similar difficulty arises at the operator-learning level: Li et al. \cite{li2024physics} show that combining data supervision with PDE constraints converts operator learning into a semi-supervised problem, stabilizing optimization where purely physics-driven training fails. We adopt an analogous strategy at the instance level through \textit{Curriculum Learning} \cite{bengio2009curriculum}: sparse synthetic reference values guide the network toward a valid basin of attraction during early training, after which the data weight is annealed so that physics constraints dominate. We introduce an auxiliary supervision loss $\lambda_{\text{data}} \mathcal{L}_{\text{data}}$ with a linearly decaying weight:
\begin{equation}
    \lambda_{\text{data}}(t) = \lambda_{\text{init}} \cdot \max\left(0, 1 - \frac{t}{T_{\text{decay}}}\right) + \lambda_{\text{final}}.
\end{equation}
Empirically, setting $\lambda_{\text{init}} = 200$ and $T_{\text{decay}} = 0.8T$ effectively relaxes the optimization problem: the network first learns the coarse solution shape via strong supervision, after which the physics constraints gradually take over to enforce conservation laws.

### Para 18 — Two-Phase Training

For problems with non-circular or complex interfaces, we further decouple the learning process into a "Topology-First" two-phase regime. In \textbf{Phase 1} ($t < 0.3T$), the training is purely data-driven ($\lambda_{\text{data}} \gg \lambda_{\text{pde}}$), forcing the network to resolve the interface geometry and solution continuity. In \textbf{Phase 2}, we restore the full physics loss to fine-tune the solution, ensuring exact satisfaction of the flux jumps. This strategy acts as a numerical homotopy method, continuously deforming the loss landscape from a convex supervised problem to the target physics-constrained problem, reducing the final error by an order of magnitude.
