"""
Neural Network Architectures for G-GRN.

Architecture:
    Input[N, 4] -> GGRN_Layer(4→H) ->[GGRN_Layer(H→H) x (L-1)] -> Decoder(H→64→1) -> Output [N, 1]

Key Components:
    - DerivativeAggregator: MessagePassing layer using precomputed GFD stencils.
    - GGRN_Layer: Combines node features with spatial derivatives via MLP.
    - GGRN: The main model orchestrating the layers.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class DerivativeAggregator(MessagePassing):
    """
    Aggregates neighbor information weighted by precomputed GFD stencil coefficients.
    
    Computes: out_i = Σ_j w_ij * (x_j - x_i)
    """
    def __init__(self):
        super().__init__(aggr="add", flow="source_to_target")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features. Shape: # [N, C]
            edge_index: Graph connectivity. Shape: # [2, E]
            coeffs: Precomputed stencil coefficients. Shape: # [E, 1]

        Returns:
            Aggregated spatial derivatives. Shape: # [N, C]
        """
        # propagate internally calls message() and aggregates the results
        return self.propagate(edge_index, x=x, edge_weight=coeffs)

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """
        Constructs the message for each edge.
        x_j: Source node features. Shape: # [E, C]
        x_i: Target node features. Shape: # [E, C]
        edge_weight: Stencil coefficient for the edge. Shape: # [E, 1]
        """
        # GFD formula: w_ij * (u_j - u_i)
        return edge_weight * (x_j - x_i)  # [E, C]


class GGRN_Layer(nn.Module):
    """
    Single G-GRN processing layer.

    Computes spatial derivatives and updates node features via MLP:
        h' = MLP([h, ∂h/∂x, ∂h/∂y, Δh]) + h  (with residual if dims match)
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.aggregator = DerivativeAggregator()

        # Input to MLP is the concatenation of [h, dx, dy, lap] -> 4 * in_channels
        combined_dim = in_channels * 4
        hidden_dim = in_channels * 2
        
        # MLP gradually compresses 4C -> 2C -> C
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )
        
        # Enable residual connection only if input and output dimensions match
        self.residual = (in_channels == out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, data) -> torch.Tensor:
        """
        Args:
            x: Node features. Shape: # [N, in_channels]
            edge_index: Graph connectivity. Shape: # [2, E]
            data: PyG Data object containing coeff_dx, coeff_dy, coeff_lap.

        Returns:
            Updated node features. Shape: # [N, out_channels]
        """
        # 1. Compute spatial derivatives using precomputed GFD stencils
        feat_dx  = self.aggregator(x, edge_index, data.coeff_dx)   # [N, in_channels]
        feat_dy  = self.aggregator(x, edge_index, data.coeff_dy)   # [N, in_channels]
        feat_lap = self.aggregator(x, edge_index, data.coeff_lap)  # [N, in_channels]

        # Normalize stencil outputs to O(1) regardless of resolution
        # coeff_dx ∝ 1/h, coeff_lap ∝ 1/h² → multiply by h, h² to cancel
        h_char = data.h_char
        feat_dx  = feat_dx  * h_char        # O(1/h) * h   = O(1)
        feat_dy  = feat_dy  * h_char        # O(1/h) * h   = O(1)
        feat_lap = feat_lap * h_char ** 2   # O(1/h²) * h² = O(1)

        # 2. Concatenate: [h, ∂h/∂x, ∂h/∂y, Δh]
        combined = torch.cat([x, feat_dx, feat_dy, feat_lap], dim=1)  #[N, 4 * in_channels]

        # 3. Pass through MLP
        out = self.mlp(combined)  # [N, out_channels]
        
        # 4. Apply residual connection if applicable
        if self.residual:
            out = out + x  # [N, out_channels]
            
        return out


class GGRN(nn.Module):
    """
    Graph-based Generalized Regression Network.

    Operates directly on 4D raw features: [x, y, phi, z].
    """
    def __init__(
        self,
        hidden_channels: int = 128,
        out_channels: int = 1,
        num_layers: int = 6,
        in_channels: int = 4,  # Default to 4 for [x, y, phi, z]
    ):
        super().__init__()

        # Processor layers
        # The first layer lifts the raw input dimension (4) to the hidden dimension (C)
        # Subsequent layers process features within the hidden dimension (C -> C)
        layers = [GGRN_Layer(in_channels, hidden_channels)]
        for _ in range(num_layers - 1):
            layers.append(GGRN_Layer(hidden_channels, hidden_channels))
            
        self.layers = nn.ModuleList(layers)

        # Decoder maps hidden representations to the final scalar/vector field
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, 64), 
            nn.GELU(), 
            nn.Linear(64, out_channels)
        )

    def forward(self, data) -> torch.Tensor:
        """
        Args:
            data: PyG Data object with x, edge_index, and stencil coefficients.

        Returns:
            Predicted field values. Shape: # [N, out_channels]
        """
        h = data.x  # [N, 4]: (x, y, phi, z)
        edge_index = data.edge_index  # [2, E]

        # 1. Process through GGRN message passing layers
        for layer in self.layers:
            h = layer(h, edge_index, data)  #[N, hidden_channels]

        # 2. Decode to final output
        out = self.decoder(h)  # [N, out_channels]
        
        return out


# ============================================================================
# Quick Test Block
# ============================================================================
if __name__ == "__main__":
    from data import MMSDataGenerator, StencilCoefficientComputer
    
    # 1. Prepare dummy data
    print("Preparing data for model test...")
    gen = MMSDataGenerator(resolution=32)
    data = gen.build_graph()
    computer = StencilCoefficientComputer(max_order=2)
    data = computer.compute_stencils(data)

    # 2. Initialize model
    print("\nInitializing G-GRN model...")
    model = GGRN(hidden_channels=64, out_channels=1, num_layers=3)
    
    # 3. Forward pass
    out = model(data)
    
    print(f"\n✅ Forward pass successful!")
    print(f"  Input shape:  {data.x.shape} -> [N, 4]")
    print(f"  Output shape: {out.shape} -> [N, 1]")
    print(f"  Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 4. Backward pass test (ensure no disconnected computational graphs)
    loss = out.mean()
    loss.backward()
    print("✅ Backward pass successful! (Gradients computed without errors)")