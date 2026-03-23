# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

G-GRN (Graph-based Generalized Regression Network): A Physics-Informed Neural Network framework that combines Graph Neural Networks with precomputed GFD (Generalized Finite Differences) stencils to solve 2D heterogeneous elliptic PDEs (`−∇·(β∇u) = f`) with discontinuous coefficients across interface geometries (circular, oscillating, elliptic).

## Commands

```bash
# Run experiments (main entry point)
python run.py --case mms --config configs/case1/beta-m-p-1-10.yaml
python run.py --case oscillating --config configs/case2/config.yaml
python run.py --case elliptic --config configs/case3/config.yaml
python run.py --case convergence --config configs/convergence/config.yaml

# CLI overrides (take precedence over YAML)
python run.py --case mms --epochs 2000 --lr 0.5 --resolution 64

# Quick sanity checks (each module has __main__ tests)
python test_data.py    # data pipeline
python model.py        # forward + backward pass
python loss.py         # loss computation for Cases 1 & 3
python train.py        # L-BFGS training loop

# Run all experiments
python log.py
```

No formal test framework (pytest), linting, or packaging setup exists.

## Architecture

**Data pipeline** (`data.py`): `BasePDEDataGenerator` subclasses (`MMSDataGenerator`, `OscillatingDataGenerator`, `EllipticInterfaceDataGenerator`) build mesh graphs on [-1,1]² with refined nodes near the interface. `StencilCoefficientComputer` precomputes GFD weights (∂/∂x, ∂/∂y, Δ) via least-squares on homophilous neighbors — these replace autograd for derivative approximation.

**Model** (`model.py`): `GGRN` stacks `GGRN_Layer` modules. Each layer uses three `DerivativeAggregator` (subclass of `torch_geometric.MessagePassing`) to compute spatial derivatives via scatter operations on precomputed stencil coefficients, concatenates results, and applies MLP with residual connection. Final decoder maps to scalar u prediction.

**Loss** (`loss.py`): `ConsistentStrongFormLoss` combines four terms: PDE residual `(−Δu − f/β)²`, boundary condition, interface jump conditions `[u]=J1` and `[β∂u/∂n]=J2`, and supervised data loss (on 5% of nodes via `data_frac`).

**Training** (`train.py`): `Trainer` uses L-BFGS with cosine annealing, gradient clipping, NaN-guard (restores best model), and best-model tracking.

**Baseline** (`baseline/pinn.py`): Vanilla PINN using MLP + autograd for derivative computation. `baseline/convergence.py` runs multi-resolution convergence studies.

## Configuration

Three-layer priority: **CLI args > case YAML > `configs/default.yaml`**. The loader in `utils.py::get_args()` merges these. Case configs only specify overrides from default.

Key parameters: `resolution`, `beta_minus/beta_plus` (coefficient contrast), `hidden_channels` (128), `num_layers` (6), loss weights `w_pde/w_bc/w_jump/w_data`.

## Key Design Details

- Node features are `[x, y, φ, z]` where φ is the level-set and z=sign(φ) marks inside/outside
- GFD stencils use adaptive order reduction: order 2 (≥5 neighbors) → order 1 (≥3 neighbors)
- Case 3 (elliptic) uses 2-phase training: data-driven first, then physics-informed
- J2 loss auto-scales by max(target_j2) to prevent gradient explosion
- Ellipse interface uses midpoint normals on heterophilous edges, not node-based normals

## Dependencies

PyTorch, torch_geometric, torch_scatter, torch_cluster, NumPy, Matplotlib, PyYAML.
