"""
Unified Physics-Informed Loss Function for G-GRN.

Integrates:
    - Standard Circular Interface Loss (Case 1 & Case 2)
    - Elliptic Interface Loss with Midpoint Normal Approximation (Case 3)

Components:
    1. PDE Residual: -Δu - f/β = 0 (Normalized form)
    2. Boundary Conditions: Dirichlet BC
    3. Interface Jump Conditions: [u] = J1, [β∂u/∂n] = J2
"""

import torch
import torch.nn as nn
from torch_scatter import scatter_add


class ConsistentStrongFormLoss(nn.Module):
    """
    Unified Strong-form physics loss for heterogeneous elliptic PDEs.
    Auto-adapts to circular or elliptic interfaces based on data attributes.
    """
    def __init__(self, w_pde: float = 1.0, w_bc: float = 100.0,
                 w_jump: float = 10.0,
                 w_j1: float = None, w_j2: float = None,
                 j2_scale: float = None):
        super().__init__()
        self.w_pde = w_pde
        self.w_bc = w_bc
        self.w_jump = w_jump

        # Fallbacks for specific jump weights
        self.w_j1 = w_j1 if w_j1 is not None else w_jump
        self.w_j2 = w_j2 if w_j2 is not None else w_jump
        
        # J2 scale to prevent gradient domination from extreme beta ratios
        self.j2_scale = j2_scale

    def compute_derivatives(self, u: torch.Tensor, data) -> tuple:
        """
        Compute spatial derivatives using precomputed GFD stencils.
        Formula: ∂u/∂x|_i = Σ_j w^x_ij * (u_j - u_i)
        """
        row, col = data.edge_index
        u_diff = u[col] - u[row]  # [E, 1]

        # Weighted aggregation via scatter_add
        dx  = scatter_add(u_diff * data.coeff_dx.view(-1, 1), row, dim=0, dim_size=data.num_nodes)   # [N, 1]
        dy  = scatter_add(u_diff * data.coeff_dy.view(-1, 1), row, dim=0, dim_size=data.num_nodes)   # [N, 1]
        lap = scatter_add(u_diff * data.coeff_lap.view(-1, 1), row, dim=0, dim_size=data.num_nodes)  # [N, 1]
        return dx, dy, lap

    def compute_ellipse_normal_at_point(self, pos: torch.Tensor, a: float, b: float) -> torch.Tensor:
        """Compute ellipse outward normal at given positions."""
        x, y = pos[:, 0], pos[:, 1]
        nx = x / (a ** 2)
        ny = y / (b ** 2)
        norm = torch.sqrt(nx ** 2 + ny ** 2).clamp(min=1e-8)
        return torch.stack([nx / norm, ny / norm], dim=1)  # [N, 2]

    def compute_ellipse_j2_at_point(self, pos: torch.Tensor, a: float, b: float,
                                    c: float, beta_minus: float, beta_plus: float) -> torch.Tensor:
        """Compute J2 = 2·sqrt(x²/a⁴ + y²/b⁴)·(c·β⁺ - β⁻) at given positions."""
        x, y = pos[:, 0], pos[:, 1]
        factor = torch.sqrt((x / a ** 2) ** 2 + (y / b ** 2) ** 2).clamp(min=1e-8)
        j2_coeff = c * beta_plus - beta_minus
        return (2.0 * factor * j2_coeff).unsqueeze(-1)  # [N, 1]

    def forward(self, u_pred: torch.Tensor, data) -> tuple:
        """
        Compute total physics-informed loss and breakdown dict.
        """
        dx, dy, lap = self.compute_derivatives(u_pred, data)

        z = data.x[:, 3].view(-1, 1)  #[N, 1]
        beta = torch.where(z < 0, data.beta_minus, data.beta_plus)  # [N, 1]

        # === Term 1: PDE Residual (Normalized) ===
        res_pde = -lap - (data.source / beta)
        loss_pde = (res_pde ** 2).mean()

        # === Term 2: Boundary Conditions ===
        mask_bnd = (torch.abs(data.pos[:, 0]) > 0.99) | (torch.abs(data.pos[:, 1]) > 0.99)
        if mask_bnd.sum() > 0:
            loss_bc = (u_pred[mask_bnd] - data.y[mask_bnd]).pow(2).mean()
        else:
            loss_bc = torch.tensor(0.0, device=u_pred.device)

        # === Term 3: Interface Jump Conditions ===
        mask_hetero = (data.edge_attr.view(-1) == 1)
        
        if mask_hetero.sum() > 0:
            src, dst = data.edge_index[:, mask_hetero]
            z_dst = z[dst]
            sign = torch.sign(z_dst)  # +1 if dst outside (Ω+), -1 if dst inside (Ω-)

            # A. Value jump (J1): [u] = u+ - u-
            val_jump_pred = sign * (u_pred[dst] - u_pred[src])
            target_j1 = data.j1[src]
            loss_j1 = (val_jump_pred - target_j1).pow(2).mean()

            # B. Flux jump (J2): [β∂u/∂n]
            # Check if this is the Elliptic Case (Case 3) requiring midpoint logic
            is_elliptic = hasattr(data, 'a') and hasattr(data, 'b')

            if is_elliptic:
                # --- Case 3 Logic: Elliptic Interface Midpoint Approximation ---
                a, b = float(data.a), float(data.b)
                pos_src = data.pos[src]
                pos_dst = data.pos[dst]
                pos_mid = 0.5 * (pos_src + pos_dst)  # Midpoint of the heterophilous edge

                n_mid = self.compute_ellipse_normal_at_point(pos_mid, a, b)
                
                # Normal derivatives projected onto midpoint normal
                dn_src = dx[src] * n_mid[:, 0:1] + dy[src] * n_mid[:, 1:2]
                dn_dst = dx[dst] * n_mid[:, 0:1] + dy[dst] * n_mid[:, 1:2]

                flux_src = beta[src] * dn_src
                flux_dst = beta[dst] * dn_dst
                flux_jump_pred = sign * (flux_dst - flux_src)

                # Hardcoded c=0.5 preserves exact original case 3 math
                c = getattr(data, 'c', 0.5) 
                target_j2 = self.compute_ellipse_j2_at_point(
                    pos_mid, a, b, c, float(data.beta_minus), float(data.beta_plus)
                )
                loss_j2 = (flux_jump_pred - target_j2).pow(2).mean()

            else:
                # --- Case 1 & 2 Logic: Circular Interface Node Approximation ---
                r_src = torch.norm(data.pos[src], dim=1, keepdim=True).clamp(min=1e-8)
                r_dst = torch.norm(data.pos[dst], dim=1, keepdim=True).clamp(min=1e-8)
                n_src = data.pos[src] / r_src  #[E_hetero, 2]
                n_dst = data.pos[dst] / r_dst  # [E_hetero, 2]

                dn_src = dx[src] * n_src[:, 0:1] + dy[src] * n_src[:, 1:2]
                dn_dst = dx[dst] * n_dst[:, 0:1] + dy[dst] * n_dst[:, 1:2]

                flux_src = beta[src] * dn_src
                flux_dst = beta[dst] * dn_dst
                flux_jump_pred = sign * (flux_dst - flux_src)
                
                target_j2 = data.j2[src]
                
                # Auto-scaling to prevent gradient domination
                j2_scale = self.j2_scale
                if j2_scale is None:
                    j2_scale = max(target_j2.abs().max().item(), 1.0)
                loss_j2 = ((flux_jump_pred - target_j2) / j2_scale).pow(2).mean()

            loss_jump = self.w_j1 * loss_j1 + self.w_j2 * loss_j2
            
        else:
            loss_j1 = torch.tensor(0.0, device=u_pred.device)
            loss_j2 = torch.tensor(0.0, device=u_pred.device)
            loss_jump = torch.tensor(0.0, device=u_pred.device)

        # Total Loss
        total_loss = self.w_pde * loss_pde + self.w_bc * loss_bc + loss_jump

        return total_loss, {
            "pde": loss_pde.item(),
            "bc": loss_bc.item(),
            "j1": loss_j1.item(),
            "j2": loss_j2.item(),
            "jump": loss_jump.item(),
        }

# ============================================================================
# Quick Test Block
# ============================================================================
if __name__ == "__main__":
    from data import MMSDataGenerator, EllipticInterfaceDataGenerator, StencilCoefficientComputer
    from model import GGRN
    
    # 1. Test Case 1 (Circular)
    print("Testing Loss on MMS (Circular) Data...")
    data_mms = StencilCoefficientComputer().compute_stencils(MMSDataGenerator(resolution=32).build_graph())
    model = GGRN(hidden_channels=64, num_layers=3)
    u_pred_mms = model(data_mms)
    
    criterion = ConsistentStrongFormLoss(w_pde=1.0, w_bc=100.0, w_jump=10.0)
    loss_mms, dict_mms = criterion(u_pred_mms, data_mms)
    print(f"✅ MMS Loss computed successfully! Total: {loss_mms.item():.4f}")
    print(f"   Breakdown: {dict_mms}")

    # 2. Test Case 3 (Elliptic)
    print("\nTesting Loss on Elliptic Data...")
    data_ell = StencilCoefficientComputer().compute_stencils(EllipticInterfaceDataGenerator().build_graph())
    u_pred_ell = model(data_ell)
    
    loss_ell, dict_ell = criterion(u_pred_ell, data_ell)
    print(f"✅ Elliptic Loss computed successfully! Total: {loss_ell.item():.4f}")
    print(f"   Breakdown: {dict_ell}")