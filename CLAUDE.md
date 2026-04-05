# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

G-GRN (Graph-based Generalized Regression Network): A Physics-Informed Neural Network framework that combines Graph Neural Networks with precomputed GFD (Generalized Finite Differences) stencils to solve 2D heterogeneous elliptic PDEs (`−∇·(β∇u) = f`) with discontinuous coefficients across interface geometries (circular, oscillating, elliptic).

**Publication target**: Computers & Mathematics with Applications or Engineering Analysis with Boundary Elements. Paper draft in `paper/` (methodology complete, experiments section in progress).

## Commands

```bash
# Run experiments (main entry point)
python run.py --case mms --config configs/case1/beta-m-p-1-10.yaml
python run.py --case mms --config configs/case1/beta-m-p-1-100.yaml
python run.py --case mms --config configs/case1/beta-m-p-10-1.yaml
python run.py --case oscillating --config configs/case2/config.yaml
python run.py --case elliptic --config configs/case3/config.yaml
python run.py --case convergence --config configs/convergence/config.yaml

# GGRN vs PINN comparison (runs both methods across resolutions)
python run.py --case convergence --config configs/case1/ggrn-pinn.yaml

# CLI overrides (take precedence over YAML)
python run.py --case mms --epochs 2000 --lr 0.5 --resolution 64

# Quick sanity checks (each module has __main__ tests)
python test_data.py    # data pipeline
python model.py        # forward + backward pass
python loss.py         # loss computation for Cases 1 & 3
python train.py        # L-BFGS training loop

# Batch run all experiments (Case 1/2/3 across resolutions)
python log.py
```

No formal test framework (pytest), linting, or packaging setup exists.

## Architecture

**Data pipeline** (`data.py`): `BasePDEDataGenerator` subclasses (`MMSDataGenerator`, `OscillatingDataGenerator`, `EllipticInterfaceDataGenerator`) build mesh graphs on [-1,1]² with refined nodes near the interface. `StencilCoefficientComputer` precomputes GFD weights (∂/∂x, ∂/∂y, Δ) via least-squares on homophilous neighbors — these replace autograd for derivative approximation. Each data object stores `h_char = 2/(resolution-1)` for stencil normalization in the model.

**Model** (`model.py`): `GGRN` stacks `GGRN_Layer` modules. Each layer uses three `DerivativeAggregator` (subclass of `torch_geometric.MessagePassing`) to compute spatial derivatives via scatter operations on precomputed stencil coefficients, **normalizes outputs by `h_char`** (dx/dy × h, lap × h²) to keep features O(1) across resolutions, concatenates results, and applies MLP with residual connection. Final decoder maps to scalar u prediction.

**Loss** (`loss.py`): `ConsistentStrongFormLoss` combines four terms: PDE residual `(−Δu − f/β)²`, boundary condition, interface jump conditions `[u]=J1` and `[β∂u/∂n]=J2`, and supervised data loss (on 5% of nodes via `data_frac`). Auto-detects circular vs elliptic interface via `hasattr(data, 'a')`.

**Training** (`train.py`): `Trainer` uses L-BFGS with cosine annealing, gradient clipping, NaN-guard (restores best model), and best-model tracking.

**Baseline** (`baseline/pinn.py`): Vanilla PINN using MLP + autograd for derivative computation. `baseline/convergence.py` runs multi-resolution convergence studies with extended metrics (MSE, Rel_L2, Rel_Linf, Abs_L2, Abs_Linf, Max_Error) matching GGRN's output format.

**Entry point** (`run.py`): Routes to data generator by `--case`, handles single-phase training (Cases 1 & 2) and 2-phase training (Case 3: data-driven → physics-informed). Outputs 3-panel PNG (Training Convergence, Exact Solution, G-GRN Prediction) + JSON metrics.

**Batch runner** (`log.py`): Loops Case 2/3 across resolutions [16, 24, 32, 64] with `num_layers=2`, `epochs=2000`. Case 1 section commented out (already completed).

## Experiment Results

### Case 1: MMS Circular Interface — COMPLETE
- 3 β contrasts (1:10, 1:100, 10:1) × 4 resolutions (16, 24, 32, 64)
- `num_layers=2`, `epochs=1000`
- Rel_L2 at r=64: 6.2e-4 ~ 7.3e-4 across all β contrasts
- Results: `results/case1/beta-m-p-{m}-{p}-r-{res}/`

### Case 2: Oscillating Angular Solution — COMPLETE
- m=3, ε=0.3, β⁻=1, β⁺=10, 4 resolutions
- `num_layers=2`, `epochs=2000`
- Rel_L2: 2.33e-3 (r=16) → 1.14e-3 (r=64), good convergence
- Results: `results/case2/r-{res}/`

### Case 3: Elliptic Interface — COMPLETE (needs tuning)
- a=0.6, b=0.4, β⁻=1, β⁺=10, 4 resolutions, 2-phase training
- `num_layers=2`, `epochs=2000`
- **Known issue**: r=16 (Rel_L2=2.85e-3) outperforms r=24/32 (~1.8e-2). Non-monotone convergence suggests 2-phase training hyperparameters need tuning for larger grids (epoch allocation, LR scheduling).
- Results: `results/case3/r-{res}/`

### PINN Baseline Convergence — COMPLETE
- 3 β contrasts × 4 resolutions
- GGRN outperforms PINN: ~1.7x lower Rel_L2, ~2.5x faster training at r=64
- Results: `results/convergence/beta-m-p-{m}-{p}/pinn_baseline_results.csv`

### Ablation: w_j2 sensitivity — COMPLETE
- w_j2=0.1 optimal; higher values degrade rapidly
- Results: `results/fix_stencil/` and `results/fix_stencil_final/`

## Configuration

Three-layer priority: **CLI args > case YAML > `configs/default.yaml`**. The loader in `utils.py::get_args()` merges these. Case configs only specify overrides from default.

Key parameters: `resolution`, `beta_minus/beta_plus` (coefficient contrast), `hidden_channels` (128), `num_layers` (default 6, experiments use 2), loss weights `w_pde/w_bc/w_jump/w_data`, `w_j2` (default 0.1, low because stencil-based flux derivatives are noisy at coarse grids).

## Key Design Details

- Node features are `[x, y, φ, z]` where φ is the level-set and z=sign(φ) marks inside/outside
- GFD stencils use adaptive order reduction: order 2 (≥5 neighbors) → order 1 (≥3 neighbors)
- Stencil outputs are resolution-normalized in `GGRN_Layer`: dx/dy × h_char, lap × h_char² → O(1) features
- Case 3 (elliptic) uses 2-phase training: Phase 1 (1/3 epochs, data-driven only), Phase 2 (2/3 epochs, physics-informed with w_j1/w_j2 from config)
- J2 loss auto-scales by max(target_j2) to prevent gradient explosion; `w_j2=0.1` by default (noisy flux stencils)
- Ellipse interface uses midpoint normals on heterophilous edges, not node-based normals
- `num_layers=2` found sufficient via ablation (2-3 layers show similar performance)

## Paper Status

- `paper/main.tex`: Full methodology written (Problem Formulation, Graph Construction, GFD Stencils, Network Architecture, Loss, Training Protocols). Experiments section ~95% commented out (placeholder structure).
- `paper/2026-3-12-third-modified.tex`: Earlier revision with partial results.
- `paper/reference.bib`: 22 citations covering PINNs, GNNs, GFD, spectral bias.
- **Still needed**: convergence order analysis (log-log slopes), error heatmaps, experiments section writing, abstract.

## Dependencies

PyTorch, torch_geometric, torch_scatter, torch_cluster, NumPy, Matplotlib, PyYAML.

## Paradigms for Avoiding AI-Generated Academic Text

### 1. 程式化句首标记（Formulaic Sentence Openers）
**特征表现**：
AI 在段落衔接时大量使用两类程式化句首标记：（a）时间与递进副词，如 **"Subsequently"、"More recently"、"Furthermore"**，将文献按时间顺序线性罗列；（b）评价性元话语填充词，如 **"Notably,"、"Importantly,"、"It is worth noting that"、"Interestingly,"**，以元评论代替实质内容。两者共同特征是占据句首位置却不承载新信息。

**规避策略（Humanized Strategy）**：
问题不在于副词本身，而在于**缺乏明确的组织原则（organizing principle）**。学术写作应优先基于**方法、理论框架或问题设定**对文献进行分组与整合，而非仅按时间推进。时间信息可以作为补充，而不应成为主导结构。对于元话语填充词，如果一个观点确实重要，其重要性应由**内容本身和上下文位置**体现，而非由 "Notably" 这类标签声明。直接陈述事实，让读者自行判断其意义。

---

### 2. 词汇膨胀与信息稀释（Lexical Inflation）
**特征表现**：
AI 倾向在两个层面稀释信息密度：（a）**短语层面**，使用模式化长表达以营造”学术感”，例如 “constitute a fundamental class of problems”、”have attracted attention for their ability to”，形式复杂但信息增量有限；（b）**动词层面**，系统性地用华丽同义词替代简单动词，如 “leverage” 替代 “use”，”utilize” 替代 “apply”，”facilitate” 替代 “enable”，制造虚假的正式感。

**规避策略（Humanized Strategy）**：
优化的关键不是一味变短，而是**提高信息密度（information density）**。应删除不增加语义内容的形式化表达，并优先使用更直接的动词结构。例如：
- 用 **”are fundamental to”** 替代冗长名词化结构
- 用 **”such as”** 替代冗余引导短语
- 用 **”use”** 替代 “utilize/leverage/employ”，除非语境确实需要区分语义（如 “employ” 强调有目的的部署）

同时保留那些**能够压缩复杂信息的必要结构**，而非简单追求句子简短。

---

### 3. 句式过度对称与节奏单一
**特征表现**：
AI 文本在句法上往往呈现高度均匀的主谓宾结构，句长与节奏趋于一致，缺乏人类写作中自然形成的节奏变化。

**规避策略（Humanized Strategy）**：
与其刻意“打破对称”，更重要的是让**句法结构服务于信息结构**。不同语义功能自然对应不同句式，例如：
- 定义与结论 → 简洁句  
- 限制与让步 → 从句或插入结构  
- 对比关系 → 结构变化  

可以适度使用前置状语、让步从句等手段，但核心目标是**体现语义层级，而非制造表面变化**。

---

### 4. 句际衔接机械化（缺乏深层语义连接）
**特征表现**：
AI 常依赖 therefore、moreover 等显性连接词拼接句子，但句与句之间缺乏实质性的语义延续，导致逻辑显得松散。

**规避策略（Humanized Strategy）**：
应通过**语义连续性（cohesion）**来实现衔接，而非仅依赖连接副词：
- **内容重叠（overlap）**：重复或延续关键词汇。
- **代词与替代（pro-form）**：如 `this method`, `these systems`, `it` 等。
- **句法延续**：使用分号或关系从句连接紧密相关的命题。
- **功能性信号词（signalling language）**：根据具体逻辑关系（因果、对比、让步）精准选择，而非重复使用通用连接词。

核心在于消除句际之间的”语义断层”，而不是简单增加连接标记。

---

### 5. 滥用破折号（Em Dash Overuse）
**特征表现**：
AI 生成文本中频繁使用破折号（ — 或 -- ）进行插入、解释或强调，例如 “This approach — which combines X and Y — achieves...” 或 “The key challenge — efficiently handling Z — remains open.”。这种模式在 GPT 风格文本中尤为显著，已成为 AI 生成内容的强识别特征。

**规避策略（Humanized Strategy）**：
在学术写作中**完全避免使用破折号**。破折号所承担的语义功能均有更正式、更符合学术规范的替代手段：
- **插入说明** → 使用括号 `()` 或 `which` 引导的非限制性定语从句
- **强调或转折** → 使用独立句子，配合适当的逻辑连接
- **列举或解释** → 使用冒号 `:` 或 `namely`, `specifically` 等词

破折号在学术论文中本身就不常见；其高频出现几乎直接暴露文本的 AI 生成痕迹。

---

### 6. 过度对冲（Hedging Overload）
**特征表现**：
AI 倾向堆叠不确定性标记以规避断言风险，例如 "may potentially contribute to"、"could possibly be attributed to"、"it seems likely that"。单个句子中出现多个对冲词（may, potentially, possibly, likely）导致论断力度被逐层稀释，读者无法判断作者对自身结论的信心程度。

**规避策略（Humanized Strategy）**：
每个命题只使用**一个精确的认识情态标记（epistemic marker）**，并根据证据强度选择合适的级别：
- **强证据** → 直接陈述，不加对冲（"X reduces Y"）
- **中等证据** → 单个情态动词（"X may reduce Y"）
- **弱证据/推测** → 明确标注证据来源（"Preliminary results suggest that X reduces Y"）

关键原则：**对冲的精度比数量重要**。"may" 和 "might" 在认识情态上有细微区别（may 表示合理可能，might 表示更远的假设），应根据实际证据水平选择，而非随意堆叠。