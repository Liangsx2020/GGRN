"""
Vanilla PINN Baseline Model and Loss.
Uses Automatic Differentiation (Autograd) instead of GFD stencils.
"""
import torch
import torch.nn as nn

class VanillaPINN(nn.Module):
    """Standard PINN baseline: MLP mapping [x, y, phi, z] -> u"""
    def __init__(self, hidden_channels: int = 128, out_channels: int = 1, num_layers: int = 6):
        super().__init__()
        layers =[nn.Linear(4, hidden_channels), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_channels, hidden_channels), nn.Tanh()])
        layers.append(nn.Linear(hidden_channels, out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, data) -> torch.Tensor:
        # Crucial for Autograd: input features must require gradient
        data.x.requires_grad_(True)
        return self.net(data.x)  # [N, 1]


class PINNStrongFormLoss(nn.Module):
    """
    Physics loss using Autograd for derivatives.
    Matches ConsistentStrongFormLoss logic to ensure fair comparison.
    """
    def __init__(self, w_pde: float = 1.0, w_bc: float = 200.0,
                 w_jump: float = 10.0, w_data: float = 0.0):
        super().__init__()
        self.w_pde = w_pde
        self.w_bc = w_bc
        self.w_jump = w_jump
        self.w_data = w_data

    def compute_autograd_derivatives(self, u: torch.Tensor, x: torch.Tensor):
        """Compute u_x, u_y, and Laplacian using torch.autograd"""
        # First derivatives
        grad_u = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]

        # Second derivatives (Laplacian)
        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0][:, 0:1]
        
        u_yy = torch.autograd.grad(
            u_y, x, grad_outputs=torch.ones_like(u_y),
            create_graph=True, retain_graph=True
        )[0][:, 1:2]

        return u_x, u_y, u_xx + u_yy

    def forward(self, u_pred: torch.Tensor, data) -> tuple:
        dx, dy, lap = self.compute_autograd_derivatives(u_pred, data.x)

        z = data.x[:, 3].view(-1, 1)
        beta = torch.where(z < 0, data.beta_minus, data.beta_plus)

        # 1. PDE Residual
        res_pde = -lap - (data.source / beta)
        loss_pde = (res_pde ** 2).mean()

        # 2. BC Loss
        mask_bnd = (torch.abs(data.pos[:, 0]) > 0.99) | (torch.abs(data.pos[:, 1]) > 0.99)
        loss_bc = (u_pred[mask_bnd] - data.y[mask_bnd]).pow(2).mean() if mask_bnd.sum() > 0 else torch.tensor(0.0, device=u_pred.device)

        # 3. Jump Loss (Circular Interface Logic)
        mask_hetero = (data.edge_attr.view(-1) == 1)
        if mask_hetero.sum() > 0:
            src, dst = data.edge_index[:, mask_hetero]
            sign = torch.sign(z[dst])

            # J1
            loss_j1 = (sign * (u_pred[dst] - u_pred[src]) - data.j1[src]).pow(2).mean()

            # J2
            r_src = torch.norm(data.pos[src], dim=1, keepdim=True).clamp(min=1e-8)
            r_dst = torch.norm(data.pos[dst], dim=1, keepdim=True).clamp(min=1e-8)
            n_src, n_dst = data.pos[src] / r_src, data.pos[dst] / r_dst

            dn_src = dx[src] * n_src[:, 0:1] + dy[src] * n_src[:, 1:2]
            dn_dst = dx[dst] * n_dst[:, 0:1] + dy[dst] * n_dst[:, 1:2]
            flux_jump_pred = sign * (beta[dst] * dn_dst - beta[src] * dn_src)
            
            j2_scale = max(data.j2[src].abs().max().item(), 1.0)
            loss_j2 = ((flux_jump_pred - data.j2[src]) / j2_scale).pow(2).mean()

            loss_jump = loss_j1 + loss_j2
        else:
            loss_jump = torch.tensor(0.0, device=u_pred.device)

        # 4. Data Loss
        loss_data = (u_pred - data.y).pow(2).mean()

        total_loss = self.w_pde * loss_pde + self.w_bc * loss_bc + self.w_jump * loss_jump + self.w_data * loss_data

        return total_loss, {"pde": loss_pde.item(), "bc": loss_bc.item(), "jump": loss_jump.item(), "data": loss_data.item()}