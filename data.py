"""
Data Generation and Graph Construction Pipeline for G-GRN.

Consolidates:
- Case 1 (MMS: Circular Interface)
- Case 2 (Oscillating Angular Solution)
- Case 3 (Elliptic Interface)
- GFD Stencil Precomputation
"""
import torch
import numpy as np
from torch_geometric.data import Data
from torch_cluster import radius_graph


# ============================================================================
# Base Data Generator
# ============================================================================

class BasePDEDataGenerator:
    """
    Abstract Base Class for generating mesh nodes, constructing the radius graph,
    and computing basic features (phi, z). Derived classes must implement the 
    specific PDE math (exact solution, source term, jump conditions).
    """
    def __init__(self, resolution: int = 32, 
                 beta_minus: float = 1.0, beta_plus: float = 10.0,
                 refine_interface: bool = True, refine_layers: int = 2,
                 refine_density: int = 64, refine_width: float = 0.1,
                 data_frac: float = 0.05):  # <-- 新增 data_frac
        self.resolution = resolution
        self.beta_minus = beta_minus
        self.beta_plus = beta_plus
        self.refine_interface = refine_interface
        self.refine_layers = refine_layers
        self.refine_density = refine_density
        self.refine_width = refine_width
        self.data_frac = data_frac

    # --- Abstract Methods (To be implemented by subclasses) ---
    def get_level_set(self, pos: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_exact_solution(self, pos: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_source_term(self, pos: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_jump_conditions(self, pos: torch.Tensor) -> tuple:
        raise NotImplementedError

    def generate_refined_nodes(self) -> torch.Tensor:
        """Return refined node coordinates near the interface, or None."""
        return None

    def post_process_data(self, data: Data) -> Data:
        """Hook to append case-specific attributes to the PyG Data object."""
        return data

    # --- Common Core Logic (Do not modify logic here) ---
    def generate_mesh_nodes(self) -> torch.Tensor:
        """Generates base uniform grid + optional interface refinement."""
        # 1. Base uniform grid on [-1, 1]^2
        lin = torch.linspace(-1, 1, self.resolution)
        xx, yy = torch.meshgrid(lin, lin, indexing='xy')
        base_pos = torch.stack([xx.flatten(), yy.flatten()], dim=1)  #[N_base, 2]

        if not self.refine_interface:
            return base_pos

        # 2. Interface refinement points
        refined_pos = self.generate_refined_nodes()
        
        if refined_pos is not None and len(refined_pos) > 0:
            return torch.cat([base_pos, refined_pos], dim=0)
        return base_pos

    def build_graph(self, r_connect: float = None, verbose: bool = False) -> Data:
        """Constructs PyG Data object with node/edge features and targets."""
        if r_connect is None:
            r_connect = 4.5 / self.resolution

        pos = self.generate_mesh_nodes()  #[N, 2]
        n_nodes = pos.size(0)

        # Level-set phi and region indicator z = sign(phi)
        phi = self.get_level_set(pos)  # [N]
        z = torch.sign(phi)
        z = torch.where(z == 0, torch.ones_like(z), z)  # Handle phi=0 edge case

        # Node features: [x, y, phi, z] -> [N, 4]
        x_feat = torch.cat([pos, phi.unsqueeze(-1), z.unsqueeze(-1)], dim=1)

        # Targets
        u_exact = self.get_exact_solution(pos)  #[N, 1]
        f_source = self.get_source_term(pos)    #[N, 1]
        j1_target, j2_target = self.get_jump_conditions(pos) # [N, 1], [N, 1]

        # Edge construction via radius graph
        edge_index = radius_graph(pos, r=r_connect, loop=False)  #[2, E]

        # Heterophilous edge detection: z[src] != z[dst]
        is_hetero = (z[edge_index[0]] != z[edge_index[1]]).float()  # [E]
        edge_attr = is_hetero.unsqueeze(-1)  #[E, 1]

        # Statistics
        if verbose:
            n_edges = edge_index.size(1)
            n_hetero_edges = int(is_hetero.sum().item())
            n_homo_edges = n_edges - n_hetero_edges
            n_inside = int((z < 0).sum().item())
            print(f"[{self.__class__.__name__} Graph Stats]")
            print(f"  Nodes: {n_nodes} (inside: {n_inside}, outside: {n_nodes - n_inside})")
            print(f"  Edges: {n_edges} (homo: {n_homo_edges}, hetero: {n_hetero_edges})")
            if n_hetero_edges == 0:
                print("[Warning] No heterophilous edges detected!")

        data = Data(
            x=x_feat, y=u_exact, pos=pos,
            edge_index=edge_index, edge_attr=edge_attr,
            source=f_source, j1=j1_target, j2=j2_target,
            phi=phi, beta_minus=self.beta_minus, beta_plus=self.beta_plus
        )

        # --- 将 5% 半监督机制“焊死”在图结构中 ---
        if self.data_frac > 0:
            n_data = max(1, int(n_nodes * self.data_frac))
            # 随机选取 5% 的节点作为监督信号传感器
            data.train_idx = torch.randperm(n_nodes)[:n_data]
        else:
            data.train_idx = None

        # Store characteristic mesh spacing for stencil normalization in model
        data.h_char = 2.0 / (self.resolution - 1)

        return self.post_process_data(data)


# ============================================================================
# Case 1: Standard MMS Verification (Circular Interface)
# ============================================================================

class MMSDataGenerator(BasePDEDataGenerator):
    """Case 1: Circular interface r=R. u = r^2 (inside), 0.5r^2 + 1 (outside)."""
    
    def __init__(self, resolution: int = 32, R: float = 0.5, **kwargs):
        super().__init__(resolution=resolution, **kwargs)
        self.R = R

    def get_level_set(self, pos: torch.Tensor) -> torch.Tensor:
        r_sq = (pos ** 2).sum(dim=1)
        return r_sq - self.R ** 2

    def get_exact_solution(self, pos: torch.Tensor) -> torch.Tensor:
        r_sq = (pos ** 2).sum(dim=1)
        mask_inside = r_sq < self.R ** 2
        u = torch.where(mask_inside, r_sq, 0.5 * r_sq + 1.0)
        return u.unsqueeze(-1)

    def get_source_term(self, pos: torch.Tensor) -> torch.Tensor:
        r_sq = (pos ** 2).sum(dim=1)
        mask_inside = r_sq < self.R ** 2
        f = torch.where(mask_inside, -4.0 * self.beta_minus, -2.0 * self.beta_plus)
        return f.unsqueeze(-1)

    def get_jump_conditions(self, pos: torch.Tensor) -> tuple:
        n_nodes = pos.size(0)
        val_j1 = 1.0 - 0.5 * self.R ** 2
        val_j2 = self.beta_plus * self.R - self.beta_minus * 2 * self.R
        J1 = torch.full((n_nodes, 1), val_j1, device=pos.device)
        J2 = torch.full((n_nodes, 1), val_j2, device=pos.device)
        return J1, J2

    def generate_refined_nodes(self) -> torch.Tensor:
        theta = torch.linspace(0, 2 * torch.pi, self.refine_density + 1)[:-1]
        refined_points =[]
        half_width = self.refine_width / 2

        for i in range(1, self.refine_layers + 1):
            offset = half_width * i / self.refine_layers
            
            # Inner layer
            r_inner = self.R - offset
            if r_inner > 0:
                x_inner, y_inner = r_inner * torch.cos(theta), r_inner * torch.sin(theta)
                refined_points.append(torch.stack([x_inner, y_inner], dim=1))

            # Outer layer
            r_outer = self.R + offset
            if r_outer < 1.0:
                x_outer, y_outer = r_outer * torch.cos(theta), r_outer * torch.sin(theta)
                refined_points.append(torch.stack([x_outer, y_outer], dim=1))

        if refined_points:
            return torch.cat(refined_points, dim=0)
        return None


# ============================================================================
# Case 2: Oscillating Angular Solution
# ============================================================================

class OscillatingDataGenerator(MMSDataGenerator):
    """Case 2: Circular interface with angular dependence cos(mθ)."""

    def __init__(self, m: int = 3, epsilon: float = 0.3, **kwargs):
        # Inherits circular refinement from MMSDataGenerator
        super().__init__(**kwargs)
        self.m = m
        self.epsilon = epsilon

    def get_theta(self, pos: torch.Tensor) -> torch.Tensor:
        return torch.atan2(pos[:, 1], pos[:, 0])

    def get_exact_solution(self, pos: torch.Tensor) -> torch.Tensor:
        r_sq = (pos ** 2).sum(dim=1)
        theta = self.get_theta(pos)
        angular = 1.0 + self.epsilon * torch.cos(self.m * theta)
        mask_inside = r_sq < self.R ** 2
        
        u_inner = r_sq * angular
        u_outer = 0.5 * r_sq * angular + 1.0
        u = torch.where(mask_inside, u_inner, u_outer)
        return u.unsqueeze(-1)

    def get_source_term(self, pos: torch.Tensor) -> torch.Tensor:
        theta = self.get_theta(pos)
        r_sq = (pos ** 2).sum(dim=1)
        mask_inside = r_sq < self.R ** 2

        cos_term = self.epsilon * (4 - self.m ** 2) * torch.cos(self.m * theta)
        lap_inner = 4.0 + cos_term
        lap_outer = 2.0 + 0.5 * cos_term

        f = torch.where(mask_inside, -self.beta_minus * lap_inner, -self.beta_plus * lap_outer)
        return f.unsqueeze(-1)

    def get_jump_conditions(self, pos: torch.Tensor) -> tuple:
        theta = self.get_theta(pos)
        angular = 1.0 + self.epsilon * torch.cos(self.m * theta)
        J1 = 1.0 - 0.5 * self.R ** 2 * angular
        J2 = self.R * angular * (self.beta_plus - 2.0 * self.beta_minus)
        return J1.unsqueeze(-1), J2.unsqueeze(-1)

    def post_process_data(self, data: Data) -> Data:
        data.m = self.m
        data.epsilon = self.epsilon
        return data


# ============================================================================
# Case 3: Elliptic Interface
# ============================================================================

class EllipticInterfaceDataGenerator(BasePDEDataGenerator):
    """Case 3: Elliptic Interface geometry."""

    def __init__(self, a: float = 0.6, b: float = 0.4, 
                 c: float = 0.5, d: float = 0.625, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        # Adjust default density for ellipse if not explicitly provided
        if 'refine_density' not in kwargs:
            self.refine_density = 80

    def get_level_set(self, pos: torch.Tensor) -> torch.Tensor:
        x, y = pos[:, 0], pos[:, 1]
        return (x / self.a) ** 2 + (y / self.b) ** 2 - 1.0

    def get_normalized_coords(self, pos: torch.Tensor) -> torch.Tensor:
        x, y = pos[:, 0], pos[:, 1]
        return (x / self.a) ** 2 + (y / self.b) ** 2

    def get_exact_solution(self, pos: torch.Tensor) -> torch.Tensor:
        xi = self.get_normalized_coords(pos)
        phi = self.get_level_set(pos)
        mask_inside = phi < 0
        u = torch.where(mask_inside, xi, self.c * xi + self.d)
        return u.unsqueeze(-1)

    def get_source_term(self, pos: torch.Tensor) -> torch.Tensor:
        phi = self.get_level_set(pos)
        mask_inside = phi < 0
        lap_const = 2.0 / (self.a ** 2) + 2.0 / (self.b ** 2)
        f = torch.where(mask_inside,
                        -self.beta_minus * lap_const * torch.ones_like(phi),
                        -self.beta_plus * self.c * lap_const * torch.ones_like(phi))
        return f.unsqueeze(-1)

    def get_normal_derivative_factor(self, pos: torch.Tensor) -> torch.Tensor:
        x, y = pos[:, 0], pos[:, 1]
        factor = torch.sqrt((x / self.a ** 2) ** 2 + (y / self.b ** 2) ** 2)
        return factor.clamp(min=1e-8)

    def get_jump_conditions(self, pos: torch.Tensor) -> tuple:
        n_nodes = pos.size(0)
        j1_val = self.c + self.d - 1.0
        J1 = torch.full((n_nodes, 1), j1_val, device=pos.device)
        
        factor = self.get_normal_derivative_factor(pos)
        j2_coeff = self.c * self.beta_plus - self.beta_minus
        J2 = 2.0 * factor * j2_coeff
        return J1, J2.unsqueeze(-1)

    def get_ellipse_normal(self, pos: torch.Tensor) -> torch.Tensor:
        x, y = pos[:, 0], pos[:, 1]
        nx = x / (self.a ** 2)
        ny = y / (self.b ** 2)
        norm = torch.sqrt(nx ** 2 + ny ** 2).clamp(min=1e-8)
        return torch.stack([nx / norm, ny / norm], dim=1)

    def generate_refined_nodes(self) -> torch.Tensor:
        t = torch.linspace(0, 2 * torch.pi, self.refine_density + 1)[:-1]
        refined_points =[]
        half_width = self.refine_width / 2

        for i in range(1, self.refine_layers + 1):
            scale_offset = half_width * i / self.refine_layers

            # Inner ellipse
            scale_inner = 1.0 - scale_offset / min(self.a, self.b)
            if scale_inner > 0:
                x_in, y_in = self.a * scale_inner * torch.cos(t), self.b * scale_inner * torch.sin(t)
                refined_points.append(torch.stack([x_in, y_in], dim=1))

            # Outer ellipse
            scale_outer = 1.0 + scale_offset / min(self.a, self.b)
            max_extent = max(self.a, self.b) * scale_outer
            if max_extent < 1.0:
                x_out, y_out = self.a * scale_outer * torch.cos(t), self.b * scale_outer * torch.sin(t)
                refined_points.append(torch.stack([x_out, y_out], dim=1))

        if refined_points:
            return torch.cat(refined_points, dim=0)
        return None

    def post_process_data(self, data: Data) -> Data:
        data.ellipse_normal = self.get_ellipse_normal(data.pos)
        data.a = self.a
        data.b = self.b
        return data


# ============================================================================
# Stencil Coefficient Computer (GFD)
# ============================================================================

class StencilCoefficientComputer:
    """Computes GFD stencil coefficients with adaptive order reduction."""

    def __init__(self, max_order: int = 2):
        self.max_order = max_order

    def _solve_least_squares(self, neighbors: list, pos_i: np.ndarray, order: int) -> tuple:
        """Robust GFD with Local Coordinate Scaling to prevent ill-conditioning at fine grids."""
        dx_list, dy_list, valid_indices = [], [],[]

        # 1. 收集局部坐标差
        for _, e_idx, p_j in neighbors:
            dx_list.append(p_j[0] - pos_i[0])
            dy_list.append(p_j[1] - pos_i[1])
            valid_indices.append(e_idx)

        dx_arr = np.array(dx_list)
        dy_arr = np.array(dy_list)

        # 2. 局部缩放因子 (Local Scaling Factor)
        # 防止网格过密导致 dx^2 变得极小，引发伪逆计算时的条件数爆炸
        distances = np.sqrt(dx_arr**2 + dy_arr**2)
        h_scale = np.max(distances)
        if h_scale < 1e-8:
            return None, None

        # 归一化到 O(1)
        dx_norm = dx_arr / h_scale
        dy_norm = dy_arr / h_scale

        # 3. 构建良态矩阵 (Well-conditioned Matrix)
        A_norm = np.column_stack([dx_norm, dy_norm])
        if order == 2:
            A_norm = np.column_stack([
                A_norm, 
                0.5 * dx_norm**2, 
                0.5 * dy_norm**2, 
                dx_norm * dy_norm
            ])

        # 4. 对归一化矩阵求逆 (因为矩阵良态，可以安全使用极小的 rcond)
        try:
            A_pinv_norm = np.linalg.pinv(A_norm, rcond=1e-6)
        except np.linalg.LinAlgError:
            return None, None

        # 5. 逆向还原真实物理权重
        A_pinv = np.zeros_like(A_pinv_norm)
        A_pinv[0, :] = A_pinv_norm[0, :] / h_scale          # w_dx
        A_pinv[1, :] = A_pinv_norm[1, :] / h_scale          # w_dy
        if order == 2:
            A_pinv[2, :] = A_pinv_norm[2, :] / (h_scale**2) # w_dxx
            A_pinv[3, :] = A_pinv_norm[3, :] / (h_scale**2) # w_dyy
            A_pinv[4, :] = A_pinv_norm[4, :] / (h_scale**2) # w_dxy

        return A_pinv, valid_indices

    def compute_stencils(self, data: Data, verbose: bool = True) -> Data:
        pos = data.pos.numpy()
        edge_index = data.edge_index.numpy()
        edge_attr = data.edge_attr.numpy().flatten()

        num_nodes = pos.shape[0]
        num_edges = edge_index.shape[1]

        coeff_dx = np.zeros(num_edges, dtype=np.float32)
        coeff_dy = np.zeros(num_edges, dtype=np.float32)
        coeff_lap = np.zeros(num_edges, dtype=np.float32)

        # Build adjacency list with homophilous edges only (edge_attr == 0)
        adj_list = [[] for _ in range(num_nodes)]
        for e_idx in range(num_edges):
            src, dst = edge_index[0, e_idx], edge_index[1, e_idx]
            if edge_attr[e_idx] == 0:
                adj_list[src].append((dst, e_idx, pos[dst]))

        order_2_count = order_1_count = failed_count = 0

        for i in range(num_nodes):
            neighbors = adj_list[i]
            pos_i = pos[i]
            num_neigh = len(neighbors)
            success = False

            # Try Order 2 (needs >= 5 neighbors)
            if self.max_order >= 2 and num_neigh >= 5:
                A_pinv, indices = self._solve_least_squares(neighbors, pos_i, order=2)
                if A_pinv is not None:
                    w_dx, w_dy = A_pinv[0, :], A_pinv[1, :]
                    w_lap = A_pinv[2, :] + A_pinv[3, :]  # Laplacian = dxx + dyy
                    for k, e_idx in enumerate(indices):
                        coeff_dx[e_idx] = w_dx[k]
                        coeff_dy[e_idx] = w_dy[k]
                        coeff_lap[e_idx] = w_lap[k]
                    order_2_count += 1
                    success = True

            # Fallback to Order 1 (needs >= 3 neighbors)
            if not success and num_neigh >= 3:
                A_pinv, indices = self._solve_least_squares(neighbors, pos_i, order=1)
                if A_pinv is not None:
                    w_dx, w_dy = A_pinv[0, :], A_pinv[1, :]
                    for k, e_idx in enumerate(indices):
                        coeff_dx[e_idx] = w_dx[k]
                        coeff_dy[e_idx] = w_dy[k]
                    order_1_count += 1
                    success = True

            if not success:
                failed_count += 1

        if verbose:
            print(f"[Stencil] Order2: {order_2_count}, Order1: {order_1_count}, Failed: {failed_count}")

        # Attach coefficients
        data.coeff_dx = torch.from_numpy(coeff_dx).float().unsqueeze(1)  #[E, 1]
        data.coeff_dy = torch.from_numpy(coeff_dy).float().unsqueeze(1)  # [E, 1]
        data.coeff_lap = torch.from_numpy(coeff_lap).float().unsqueeze(1)  # [E, 1]

        return data

if __name__ == "__main__":
    gen = MMSDataGenerator(resolution=64)
    data = gen.build_graph(verbose=True)
    