# Section 4: Numerical Experiments — 写作计划

> 段落级骨架。每段标注 **目的**、**要引用的数据/图表**、**关键数据点**。按顺序往每段填 LaTeX 即可。

---

## 4.1 Experimental Setup

### Para 1 — 问题与计算域

Three test problems on the domain $\Omega = [-1,1]^2$ probe different aspects of G-GRN's capability. The first, a circular interface at $R=0.5$, serves as a standard MMS baseline under three coefficient contrasts ($\beta^-/\beta^+ \in \{1/10, \, 1/100, \, 10/1\}$). Example \ref{ex2} introduces angular oscillations into the exact solution, testing whether the method can resolve high-frequency features near $\Gamma$. Example \ref{ex3} replaces the circle with an ellipse ($a=0.6$, $b=0.4$), where the varying normal direction along the interface stresses the flux jump approximation. All three cases admit closed-form solutions, enabling exact error quantification.

### Para 2 — 模型与训练配置

All experiments share a common architecture: a 2-layer G-GRN with 128 hidden channels per layer, a depth confirmed sufficient by preliminary ablation. We use L-BFGS (history size 50, strong Wolfe line search) as the optimizer; in our preliminary experiments on these interface problems, L-BFGS converged to consistently lower loss values than Adam. The learning rate follows a cosine annealing schedule from $\eta_0 = 1.0$ to $\eta_{\min} = 10^{-6}$. Four grid resolutions ($N \in \{16, 24, 32, 64\}$) are tested for each example, with Example \ref{ex1} trained for 1\,000 epochs and Examples \ref{ex2}--\ref{ex3} for 2\,000 epochs. The composite loss assigns weights $\lambda_{\text{pde}} = 1$, $\lambda_{\text{bc}} = 200$, $\lambda_{\Gamma} = 10$, and $\lambda_{\text{data}} = 1000$; the flux jump term receives a lower weight $w_{J_2} = 0.1$ to suppress noise from stencil-based flux derivatives at coarse resolutions. All computations run on a single NVIDIA RTX PRO 6000 Black Edition GPU.

### Para 3 — 评估指标定义

Prediction accuracy is measured by two norms: the relative $L^2$ error $e_{L^2} = \|u - \hat{u}\|_2 / \|u\|_2$, which quantifies global deviation, and the relative $L^\infty$ error $e_{L^\infty} = \|u - \hat{u}\|_\infty / \|u\|_\infty$, which isolates the worst-case pointwise mismatch. To characterize how these errors decay under grid refinement, we assume a power-law scaling $e_{L^2} \sim C h^p$ with grid spacing $h = 2/(N-1)$ and extract the convergence order $p$ by least-squares regression on $\log h$ versus $\log e_{L^2}$ (Figure \ref{fig:2}).

\begin{figure}
    \centering
    \includegraphics[width=0.5\linewidth]{images/convergence_ggrn.pdf}
    \caption{Convergence of G-GRN across three interface geometries. Log-log plot of relative $L^2$ error versus grid resolution $N$. The fitted slope $p$ denotes the convergence order.}
    \label{fig:2}
\end{figure}

### Para 4 — Example 1: 圆形界面 MMS 验证

For baseline verification, we consider a circular interface $\Gamma: r=R=0.5$. We construct the following piecewise exact solution: \label{ex1}
\begin{equation}
    u(\bf{x}) = \begin{cases}
        r^2, & r < R, \\
        \frac{1}{2} r^2 +1, &r>R.
    \end{cases}
\end{equation}
with $r = x^2 + y^2$. The corresponding source terms are $f = -4 \beta^{-}$ in $\Omega^{-}$ and $f = -2 \beta^{+}$ in $\Omega^{+}$. This solution imposes a value jump $[u] = 1 - \frac{1}{2}r^2$ and a flux jump $[\beta \partial_n u] = (\beta^+ - 2 \beta^{-}) r$ across the interface. We evaluate the method under three coefficient contrasts $\beta^{-}/\beta^{+} \in \{1/10, 1/100, 10/1\}$ and report the MSE, relative $L^2$, relative $L^{\infty}$, and CPU time across four grid resolutions in Table \ref{table1}. At the finest resolution ($N = 64$), the $e_{L^2}$ values for the three contrasts are $6.17 \times 10^{-4}$, $6.80 \times 10^{-4}$, and $7.26 \times 10^{-4}$, respectively; this narrow spread confirms that accuracy is largely insensitive to the magnitude of the coefficient jump, with no significant degradation even at a $1{:}100$ ratio.

\begin{table}[htbp]
    \centering
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{ccccccc}
        \toprule
        \multirow{2}{*}{Layer} & \multirow{2}{*}{$(\beta^-, \beta^+)$} & \multirow{2}{*}{Resolution} & \multicolumn{4}{c}{G-GRN} \\
        \cmidrule(lr){4-7}
        & & & MSE & Rel $L^2$ & Rel $L^\infty$ & CPU Time (s) \\ 
       \midrule
        \multirow{12}{*}{2} 
        & \multirow{4}{*}{(1, 10)} 
        & 16 & 2.12E-06 & 1.32E-03 & 1.26E-03 & 58.97 \\ 
        & & 24 & 2.06E-06 & 1.22E-03 & 1.36E-03 & 49.12 \\ 
        & & 32 & 2.01E-06 & 1.16E-03 & 1.72E-03 & 44.33 \\ 
        & & 64 & 6.02E-07 & \bf{6.17E-04} & 1.45E-03 & 46.26 \\ 
        \cmidrule{2-7}
        & \multirow{4}{*}{(1, 100)} 
        & 16 & 2.17E-06 & 1.33E-03 & 1.19E-03 & 59.86 \\ 
        & & 24 & 2.13E-06 & 1.24E-03 & 1.43E-03 & 48.92 \\ 
        & & 32 & 2.10E-06 & 1.19E-03 & 1.73E-03 & 41.11 \\ 
        & & 64 & 7.32E-07 & \bf{6.80E-04} & 1.14E-03 & 46.52 \\ 
        \cmidrule{2-7}
        & \multirow{4}{*}{(10, 1)} 
        & 16 & 2.37E-06 & 1.39E-03 & 1.31E-03 & 60.72 \\ 
        & & 24 & 2.40E-06 & 1.32E-03 & 1.40E-03 & 45.98 \\ 
        & & 32 & 2.30E-06 & 1.25E-03 & 1.72E-03 & 42.05 \\ 
        & & 64 & 8.34E-07 & \bf{7.26E-04} & 1.40E-03 & 43.21 \\ 
        \bottomrule
    \end{tabular}%
    }
     \caption{The MSE, relative $L^2$, relative $L^\infty$ errors and CPU time of G-GRN method in Example \ref{ex1} where $\beta^{-}/\beta{+} \in \{1/10, 1/100, 10/1\}$.}
    \label{table1}
\end{table}

We compare G-GRN against a standard PINN under identical problem settings (Table \ref{table2}). At $N = 64$ with $\beta^{-}/\beta^{+} = 1/10$, G-GRN achieves $e_{L^2} = 6.17 \times 10^{-4}$, compared to $1.03 \times 10^{-3}$ for the PINN, a $1.7\times$ accuracy gain. Training time also decreases from 114\,s to 46\,s, a $2.5\times$ speedup. Beyond the final accuracy, the two methods differ in their convergence trajectories: the PINN stagnates between $N = 24$ and $N = 32$, with $e_{L^2}$ shifting from $1.41 \times 10^{-3}$ to $1.42 \times 10^{-3}$, while G-GRN maintains a steady error reduction across the entire resolution range. We attribute this contrast to the higher fidelity of GFD-based derivative approximations relative to autograd; the precomputed stencils capture local differential structure more accurately, allowing the network to benefit from each increment in grid resolution.

\begin{table}[htbp]
    \centering
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{ccccccc}
        \toprule
        \multirow{2}{*}{Layer} & \multirow{2}{*}{$(\beta^-, \beta^+)$} & \multirow{2}{*}{Resolution} & \multicolumn{4}{c}{PINN} \\
        \cmidrule(lr){4-7}
        & & & MSE & Rel $L^2$ & Rel $L^\infty$ & CPU Time (s) \\ 
        \midrule
        \multirow{12}{*}{2} 
        & \multirow{4}{*}{(1, 10)} 
        & 16 & 2.52E-06 & 1.44E-03 & 1.66E-03 & 99.00 \\ 
        & & 24 & 2.76E-06 & 1.41E-03 & 1.66E-03 & 99.68 \\ 
        & & 32 & 2.99E-06 & 1.42E-03 & 2.21E-03 & 103.31 \\ 
        & & 64 & 1.67E-06 & \bf{1.03E-03} & 3.03E-03 & 113.62 \\ 
        \cmidrule{2-7}
        & \multirow{4}{*}{(1, 100)} 
        & 16 & 2.49E-06 & 1.43E-03 & 1.74E-03 & 75.28 \\ 
        & & 24 & 2.76E-06 & 1.41E-03 & 1.63E-03 & 87.35 \\ 
        & & 32 & 3.00E-06 & 1.42E-03 & 2.30E-03 & 105.84 \\ 
        & & 64 & 1.53E-06 & \bf{9.85E-04} & 2.86E-03 & 111.25 \\ 
        \cmidrule{2-7}
        & \multirow{4}{*}{(10, 1)} 
        & 16 & 2.44E-06 & 1.41E-03 & 1.81E-03 & 135.95 \\ 
        & & 24 & 2.62E-06 & 1.38E-03 & 1.63E-03 & 98.80 \\ 
        & & 32 & 2.94E-06 & 1.41E-03 & 2.42E-03 & 123.78 \\ 
        & & 64 & 1.63E-06 & \bf{1.01E-03} & 2.83E-03 & 42.83 \\ 
        \bottomrule
    \end{tabular}%
    }
    \caption{Performance of PINN for different values of $(\beta^-, \beta^+)$ at Layer 2.}
    \label{table2}
\end{table}

---

## 4.2 Case 2: Oscillating Angular Solution

### Para 1 — 问题描述

Example \ref{ex2} tests whether G-GRN can resolve high-frequency spatial features near the interface. Retaining the circular geometry $\Gamma:r = R = 0.5$ from Example \ref{ex1}, we modify the exact solution to include angular oscillations:
\begin{equation}
    u (\bf{x}) = \begin{cases}
        r^2(1 + \epsilon \cos m\theta), &r < R, \\
        \frac{1}{2} ( 1 + \epsilon \cos m\theta) + 1, &r > R.
    \end{cases}
\end{equation}
where $m = 3$, $\epsilon = 0.3$, $\beta^- = 1$, and $\beta^+ = 10$. The resulting source term $f$ acquires a $\cos (m \theta)$ dependence and becomes spatially non-uniform, which increases the angular gradient of the PDE residual near the interface relative to Example \ref{ex1}.

### Para 2 — 结果与误差分析

Figure \ref{fig:ex2_solution} shows the exact solution, the G-GRN prediction, and the pointwise absolute error at resolution $N = 64$. The predicted field reproduces the angular oscillation pattern with high fidelity. The error heatmap (Figure \ref{fig:ex2_solution}c) reveals that the largest deviations concentrate near $\Gamma$ at azimuthal angles corresponding to the amplitude peaks of $\cos(3\theta)$, while the interior domain maintains consistently low error levels.

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{images/ex2.pdf}
    \caption{Example 2: Oscillating angular solution with $m=3$, $\varepsilon=0.3$, $\beta^-=1$, $\beta^+=10$, $N=64$. The dashed line indicates the interface $\Gamma$.}
    \label{fig:ex2_solution}
\end{figure}

---

## 4.3 Case 3: Elliptic Interface

### Para 1 — 问题描述

Example \ref{ex3} replaces the circular interface with an ellipse $\Gamma: (x/a)^2 + (y/b)^2 = 1$ ($a = 0.6$, $b=0.4$, $\beta^- = 1$, $\beta^+ = 10$) to test G-GRN on an asymmetric geometry. Defining $\xi = (x/a)^2 + (y/b)^2$, we construct the exact solution as:
\begin{equation}
    u (\bf{x}) = \begin{cases}
        \xi, & \xi < 1, \\
        0.5 \xi + 0.625, & \xi >1.
    \end{cases}
\end{equation}
Because the outward normal of the ellipse varies continuously along the interface, this geometry imposes stricter requirements on the flux jump approximation $J_2 = [\beta \partial_n u]$ than the circular cases.

### Para 2 — 结果与误差分析

Figure \ref{fig:ex3_solution} displays the exact solution, the G-GRN prediction, and the pointwise absolute error at resolution $N=64$. The prediction preserves the gradient discontinuity across the elliptical interface, with an overall error level comparable to that of Example \ref{ex2}. The error heatmap (Figure \ref{fig:ex3_solution}c) shows that the maximum errors concentrate at the endpoints of the semi-major axis, where the curvature is largest; this pattern is consistent with the reduced accuracy of the midpoint normal approximation in high-curvature regions. At this resolution, the relative $L^2$ error is $1.88 \times 10^{-3}$. A joint convergence analysis of all three examples follows in Section \ref{sec:convergence}.

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{images/ex3.pdf}
    \caption{Example 3: Elliptic interface with $a=0.6$, $b=0.4$, $\beta^-=1$, $\beta^+=10$, $N=64$. The dashed line indicates the elliptic interface $\Gamma$.}
    \label{fig:ex3_solution}
\end{figure}

---

## 4.4 Convergence Analysis

### Para 1 — G-GRN 三 Case 收敛

Figure \ref{fig:2} summarizes the convergence behavior of all three test problems. Least-squares fits on the log-log data yield convergence orders of $p=0.54$ (Example \ref{ex1}), $p=0.45$ (Example \ref{ex2}), and $p=0.87$ (Example \ref{ex3}). All three cases exhibit a stable error reduction with increasing resolution. The highest rate is observed for Example \ref{ex3}, whose exact solution has the greatest regularity among the three test problems.

### Para 2 — G-GRN vs PINN 收敛对比

Figure \ref{fig:convergence_ggrn_pinn} compares the convergence of G-GRN and the standard PINN for Example \ref{ex1} ($\beta^- / \beta^+ = 1 / 10$). G-GRN achieves a convergence order of $p=0.54$, more than twice that of the PINN ($p = 0.24$). The gap is most pronounced between $N = 24$ and $N = 32$, where the PINN error stagnates ($e_{L^2}$: $1.41 \times 10^{-3} \to 1.42 \times 10^{-3}$) while G-GRN continues to reduce error monotonically. As discussed in Section \ref{sec:setup}, this sustained improvement reflects the higher fidelity of GFD stencil-based derivative approximations, which allow the network to translate finer grids into proportionally lower errors.

\begin{figure}
    \centering
    \includegraphics[width=0.5\linewidth]{images/convergence_ggrn_vs_pinn.pdf}
    \caption{Convergence comparison between G-GRN and PINN for the circular interface problem ($\beta^-/\beta^+=1/10$). The fitted slope $p$ indicates the convergence order.}
    \label{fig:convergence_ggrn_pinn}
\end{figure}

---

## 数据来源索引

| 数据 | 文件路径 |
|---|---|
| Case 1 指标 | `results/case1/beta-m-p-{m}-{p}-r-{res}/mms_metrics.json` |
| Case 2 指标 | `results/case2/r-{res}/oscillating_metrics.json` |
| Case 3 指标（单阶段） | `results/case3/r-{res}-single-phase/elliptic_metrics.json` |
| PINN 基线 | `results/convergence/beta-m-p-{m}-{p}/pinn_baseline_results.csv` |
| Case 2 可视化数据 | `matlab_plot/oscillating_r64_data.csv` |
| Case 3 可视化数据 | `matlab_plot/elliptic_r64_data.csv` |
| 收敛阶数据 | `matlab_plot/convergence_ggrn.xlsx` |
| GGRN vs PINN 数据 | `matlab_plot/convergence_ggrn_vs_pinn.xlsx` |
