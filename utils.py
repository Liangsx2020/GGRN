"""
Utility Functions for G-GRN Project.

Contains:
    - Smart Configuration Loading (YAML + CLI Overrides)
    - Reproducibility (Seed)
    - Unified 3D Visualization
"""
import torch
import numpy as np
import random
import argparse
import yaml
import os
import matplotlib.pyplot as plt


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_args() -> argparse.Namespace:
    """
    Smart Config Loader: default.yaml -> case YAML -> CLI overrides.
    Automatically parses unknown CLI arguments to override YAML keys.
    """
    parser = argparse.ArgumentParser(description="G-GRN Universal Runner")
    parser.add_argument('--case', type=str, default='mms', 
                        choices=['mms', 'oscillating', 'elliptic', 'convergence'],
                        help="Which case to run: mms, oscillating, elliptic, or convergence")
    parser.add_argument('--config', type=str, default=None, 
                        help="Path to specific YAML config")
    parser.add_argument('--data_frac', type=float, default=0.05, 
                        help="Fraction of nodes used for supervised data loss (default: 5%)")
    
    # Parse known basic args
    args, unknown = parser.parse_known_args()

    # 1. Load default.yaml (Single Source of Truth)
    default_path = os.path.join(os.path.dirname(__file__), 'configs', 'default.yaml')
    if not os.path.exists(default_path):
        raise FileNotFoundError(f"Cannot find {default_path}. Please ensure configs folder is copied.")
    
    with open(default_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 2. Override with specified case config (if provided)
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            case_cfg = yaml.safe_load(f) or {}
            cfg.update(case_cfg)

    # 3. Smart CLI Overrides (e.g., --epochs 100 --lr 0.5)
    i = 0
    while i < len(unknown):
        key = unknown[i]
        if key.startswith('--'):
            k = key[2:]
            if i + 1 < len(unknown) and not unknown[i+1].startswith('--'):
                val = unknown[i+1]
                # Auto type-casting
                if val.isdigit(): val = int(val)
                elif val.replace('.', '', 1).isdigit(): val = float(val)
                elif val.lower() == 'true': val = True
                elif val.lower() == 'false': val = False
                elif val.lower() == 'null': val = None
                
                cfg[k] = val
                i += 2
            else:
                cfg[k] = True  # Handle boolean flags like --quick
                i += 1
        else:
            i += 1

    # Add the strictly parsed args back into config
    cfg['case'] = args.case
    cfg['config'] = args.config
    cfg['data_frac'] = args.data_frac 

    return argparse.Namespace(**cfg)


def plot_results(model, data, history: dict, save_dir: str, filename: str) -> None:
    """
    Unified plotting function for all cases.
    Generates a 3-panel figure: Training Curve, Exact 3D Surface, Predicted 3D Surface.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        u_pred = model(data).cpu().numpy().flatten()
    
    u_gt = data.y.cpu().numpy().flatten()
    pos = data.pos.cpu().numpy()

    fig = plt.figure(figsize=(18, 5))

    # Panel 1: Training Curve
    ax0 = fig.add_subplot(1, 3, 1)
    if 'rel_l2' in history and history['rel_l2']:
        ax0.plot(history['rel_l2'], color='tab:blue', label='Rel L2', linewidth=1.2)
        ax0.plot(history['rel_linf'], color='tab:red', label='Rel Linf', linewidth=1.2)
        ax0.set_yscale('log')
        ax0.set_xlabel('Epochs')
        ax0.set_ylabel('Relative Error')
        ax0.set_title('Training Convergence')
        ax0.legend()
        ax0.grid(True, alpha=0.3)

    # Panel 2: Exact Solution
    ax1 = fig.add_subplot(1, 3, 2, projection='3d')
    sc1 = ax1.scatter(pos[:, 0], pos[:, 1], u_gt, c=u_gt, cmap='jet', s=1, alpha=0.6)
    fig.colorbar(sc1, ax=ax1, shrink=0.5, aspect=5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u')
    ax1.set_title('Exact Solution')

    # Panel 3: Prediction
    ax2 = fig.add_subplot(1, 3, 3, projection='3d')
    sc2 = ax2.scatter(pos[:, 0], pos[:, 1], u_pred, c=u_pred, cmap='jet', s=1, alpha=0.6)
    fig.colorbar(sc2, ax=ax2, shrink=0.5, aspect=5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u')
    ax2.set_title('G-GRN Prediction')

    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 Visualization saved to: {save_path}")