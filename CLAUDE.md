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
