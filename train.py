"""
Unified Training Engine for G-GRN.

Encapsulates the L-BFGS optimizer, closure mechanism, learning rate scheduling,
NaN-guard protection, and comprehensive evaluation metrics tracking.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from typing import Dict, Tuple, Any, Optional


class Trainer:
    """
    Universal trainer for Physics-Informed Neural Operators.
    Handles L-BFGS closure, best-model tracking, and memory-safe caching.
    """
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device

    def evaluate_metrics(self, u_pred: torch.Tensor, u_gt: torch.Tensor) -> Dict[str, float]:
        """Compute comprehensive error metrics."""
        err = (u_pred - u_gt).abs()
        mse = torch.mean(err ** 2).item()
        
        # Norms
        u_ref_norm = torch.sqrt((u_gt ** 2).sum()).item()
        u_ref_max = u_gt.abs().max().item()

        rel_l2 = torch.sqrt((err ** 2).sum()).item() / u_ref_norm if u_ref_norm > 0 else 0.0
        rel_linf = err.max().item() / u_ref_max if u_ref_max > 0 else 0.0
        
        abs_l2 = torch.sqrt((err ** 2).sum()).item()
        abs_linf = err.max().item()

        return {
            "mse": mse,
            "rel_l2": rel_l2,
            "rel_linf": rel_linf,
            "abs_l2": abs_l2,
            "abs_linf": abs_linf,
            "max_error": abs_linf
        }

    def fit(self, 
            data, 
            criterion: nn.Module, 
            epochs: int, 
            lr: float, 
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            scheduler_type: str = 'cosine',
            grad_clip: float = 0.0,
            log_interval: int = 100,
            phase_name: str = "Training") -> Tuple[Dict[str, list], Dict[str, float], float]:
        """
        Main training loop. Can be called sequentially for multi-phase training.

        Args:
            data: PyG Data object
            criterion: Loss function module
            epochs: Number of epochs
            lr: Learning rate
            optimizer_kwargs: L-BFGS parameters (max_iter, history_size, etc.)
            scheduler_type: 'cosine', 'step', or 'none'
            grad_clip: Max gradient norm (0 means disabled)
            log_interval: Epochs between print logs
            phase_name: Prefix for logging (e.g., "Phase 1")

        Returns:
            history (Dict): Trajectory of metrics per epoch
            final_metrics (Dict): Comprehensive metrics of the best model
            train_time (float): Wall-clock time in seconds
        """
        # 1. Setup Optimizer & Scheduler
        if optimizer_kwargs is None:
            optimizer_kwargs = {
                'max_iter': 20, 
                'history_size': 50, 
                'line_search_fn': 'strong_wolfe'
            }
            
        optimizer = optim.LBFGS(self.model.parameters(), lr=lr, **optimizer_kwargs)

        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.1)
        else:
            scheduler = None

        # 2. Tracking Variables
        history = {'mse': [], 'rel_l2': [], 'rel_linf':[]}
        best_mse = float('inf')
        best_state = None
        t_start = time.perf_counter()

        print(f"\n--- Starting {phase_name} ({epochs} Epochs, LR: {lr}) ---")

        # 3. Training Loop
        for epoch in range(epochs):
            self.model.train()
            cache = {}

            # Closure for L-BFGS
            def closure():
                optimizer.zero_grad()
                u_pred = self.model(data)
                
                # Check if criterion returns tuple (loss, terms) or just loss
                out = criterion(u_pred, data)
                if isinstance(out, tuple):
                    loss, terms = out
                else:
                    loss, terms = out, {}

                loss.backward()
                
                if grad_clip > 0:
                    clip_grad_norm_(self.model.parameters(), grad_clip)

                # Detach to prevent memory leaks in cache
                cache['u_pred'] = u_pred.detach()
                cache['loss'] = loss.detach()
                cache['terms'] = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in terms.items()}
                
                return loss

            # Optimizer Step
            optimizer.step(closure)
            if scheduler is not None:
                scheduler.step()

            # Retrieve cached results from the last closure execution
            loss = cache['loss']
            terms = cache['terms']
            u_pred = cache['u_pred']

            # 4. NaN Guard
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[🚨 Warning] NaN/Inf detected at epoch {epoch}. Restoring best model (MSE: {best_mse:.6e})")
                if best_state is not None:
                    self.model.load_state_dict(best_state)
                break

            # 5. Evaluate and Track Best Model
            with torch.no_grad():
                metrics = self.evaluate_metrics(u_pred, data.y)
                mse = metrics['mse']

            history['mse'].append(mse)
            history['rel_l2'].append(metrics['rel_l2'])
            history['rel_linf'].append(metrics['rel_linf'])

            # Store on CPU to save GPU memory!
            if mse < best_mse:
                best_mse = mse
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            # 6. Logging
            if epoch % log_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                log_str = f"[{phase_name}] Ep {epoch:5d} | Loss: {loss.item():.4e} | LR: {current_lr:.1e} | MSE: {mse:.4e}"
                if terms:
                    terms_str = ", ".join([f"{k}: {v:.4e}" for k, v in terms.items() if k in['pde', 'bc', 'jump', 'data']])
                    log_str += f" | {terms_str}"
                print(log_str)

        train_time = time.perf_counter() - t_start

        # 7. Restore Best Model for Final Evaluation
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        with torch.no_grad():
            final_metrics = self.evaluate_metrics(self.model(data), data.y)

        print(f"✅ {phase_name} Completed in {train_time:.2f}s | Best MSE: {final_metrics['mse']:.6e}")
        
        return history, final_metrics, train_time


# ============================================================================
# Quick Test Block
# ============================================================================
if __name__ == "__main__":
    from data import MMSDataGenerator, StencilCoefficientComputer
    from model import GGRN
    from loss import ConsistentStrongFormLoss

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Prepare Data
    gen = MMSDataGenerator(resolution=32)
    data = gen.build_graph()
    data = StencilCoefficientComputer(max_order=2).compute_stencils(data).to(device)

    # 2. Init Model & Loss
    model = GGRN(hidden_channels=64, num_layers=3)
    criterion = ConsistentStrongFormLoss(w_pde=1.0, w_bc=100.0, w_jump=10.0)

    # 3. Test Trainer
    trainer = Trainer(model, device)
    
    # Run a tiny 10-epoch test
    history, metrics, t_time = trainer.fit(
        data=data, 
        criterion=criterion, 
        epochs=10, 
        lr=1.0, 
        log_interval=2,
        phase_name="Test Training"
    )

    print("\n--- Final Test Metrics ---")
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.6e}")