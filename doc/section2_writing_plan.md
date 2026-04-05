# Section 2: Problem Formulation — 写作计划

## Para 1 — 计算域、控制方程与系数定义

This section formulates the elliptic interface problem studied throughout the paper. Let $\Omega \subset \mathbb{R}^2$ be a bounded computational domain bisected by a closed interface $\Gamma$ into an interior subdomain $\Omega^-$ and an exterior subdomain $\Omega^+$. In $\Omega^- \cup \Omega^+$, the scalar field $u(\mathbf{x})$ satisfies the steady-state diffusion equation:
\begin{equation}
    -\nabla \cdot (\beta(\mathbf{x}) \nabla u(\mathbf{x})) = f(\mathbf{x}),
\end{equation}
where $f(\mathbf{x})$ is a prescribed source term. The diffusion coefficient $\beta(\mathbf{x})$ is piecewise constant with a jump across $\Gamma$:

\begin{equation}
    \beta(\mathbf{x}) =
    \begin{cases}
        \beta^- & \text{for } \mathbf{x} \in \Omega^-, \\
        \beta^+ & \text{for } \mathbf{x} \in \Omega^+,
    \end{cases}
\end{equation}
where $\beta^-$ and $\beta^+$ are strictly positive constants. This discontinuity reduces the regularity of the exact solution: standard finite difference \cite{liszka1980finite} and finite element \cite{babuvska1970finite} methods lose accuracy near $\Gamma$ unless the mesh conforms to the interface geometry.

## Para 2 — 传输条件

Well-posedness requires two transmission conditions across $\Gamma$. Let $\mathbf{n}$ denote the unit normal pointing from $\Omega^-$ to $\Omega^+$. The first condition prescribes the jump in the solution value:
\begin{equation}
    [u]_\Gamma \coloneqq u^+|_\Gamma - u^-|_\Gamma = g_1,
\end{equation}
and the second prescribes the jump in the normal flux:
\begin{equation}
    \left[\beta \frac{\partial u}{\partial n}\right]_\Gamma \coloneqq \beta^+ \nabla u^+ \cdot \mathbf{n} - \beta^- \nabla u^- \cdot \mathbf{n} = g_2,
\end{equation}
where $u^\pm$ denote the one-sided limits from $\Omega^\pm$. In many physical settings $g_1 = g_2 = 0$ (perfect contact). When the contrast ratio $\beta^+/\beta^-$ is large, the flux condition forces an abrupt change in $\nabla u$ across $\Gamma$, producing a gradient discontinuity that is difficult to approximate with smooth basis functions.

## Para 3 — 边界条件、level-set 表示与问题总结

On the outer boundary $\partial\Omega$ we impose Dirichlet data $u = g_D$. The interface $\Gamma$ is represented implicitly by a level-set function $\phi(\mathbf{x})$ with $\phi < 0$ in $\Omega^-$, $\phi > 0$ in $\Omega^+$, and $\Gamma = \{\mathbf{x} : \phi(\mathbf{x}) = 0\}$. This representation provides a direct formula for the interface normal, $\mathbf{n} = \nabla\phi / |\nabla\phi|$. The combination of discontinuous coefficients, transmission conditions, and the resulting gradient singularity at $\Gamma$ defines the core difficulty of the elliptic interface problem. Section \ref{sec:method} introduces the G-GRN architecture designed to address these challenges.
