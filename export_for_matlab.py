"""
Export trained G-GRN predictions to CSV for MATLAB visualization.

Runs training and saves (x, y, u_exact, u_pred, abs_error) as CSV.

Usage:
    python export_for_matlab.py --case oscillating --resolution 64
    python export_for_matlab.py --case elliptic --resolution 64
"""
import os
import torch
import csv

from utils import get_args, set_seed
from data import (MMSDataGenerator, OscillatingDataGenerator,
                  EllipticInterfaceDataGenerator, StencilCoefficientComputer)
from model import GGRN
from loss import ConsistentStrongFormLoss
from train import Trainer


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 1. Data Pipeline
    if args.case == 'mms':
        gen = MMSDataGenerator(resolution=args.resolution,
                               beta_minus=args.beta_minus, beta_plus=args.beta_plus,
                               data_frac=args.data_frac)
    elif args.case == 'oscillating':
        gen = OscillatingDataGenerator(resolution=args.resolution,
                                       beta_minus=args.beta_minus, beta_plus=args.beta_plus,
                                       m=getattr(args, 'm', 3), epsilon=getattr(args, 'epsilon', 0.3),
                                       data_frac=args.data_frac)
    elif args.case == 'elliptic':
        gen = EllipticInterfaceDataGenerator(resolution=args.resolution,
                                             beta_minus=args.beta_minus, beta_plus=args.beta_plus,
                                             a=getattr(args, 'a', 0.6), b=getattr(args, 'b', 0.4),
                                             data_frac=args.data_frac)
    else:
        raise ValueError(f"Unknown case: {args.case}")

    data = gen.build_graph(r_connect=args.r_connect, verbose=True)
    data = StencilCoefficientComputer(max_order=args.max_order).compute_stencils(data)
    data = data.to(device)

    # 2. Model & Training
    model = GGRN(hidden_channels=args.hidden_channels, num_layers=args.num_layers).to(device)
    trainer = Trainer(model, device)

    criterion = ConsistentStrongFormLoss(
        w_pde=args.w_pde, w_bc=args.w_bc, w_jump=args.w_jump, w_data=args.w_data,
        w_j1=args.w_j1, w_j2=args.w_j2, j2_scale=args.j2_scale
    )
    trainer.fit(data, criterion, epochs=args.epochs, lr=args.lr,
                scheduler_type=args.scheduler, grad_clip=args.grad_clip,
                log_interval=args.log_interval, phase_name="Training")

    # 3. Export
    model.eval()
    with torch.no_grad():
        u_pred = model(data).cpu().numpy().flatten()

    u_exact = data.y.cpu().numpy().flatten()
    pos = data.pos.cpu().numpy()

    abs_error = [abs(e - p) for e, p in zip(u_exact, u_pred)]

    out_dir = 'matlab_plot'
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'{args.case}_r{args.resolution}_data.csv')

    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'u_exact', 'u_pred', 'abs_error'])
        for i in range(len(u_exact)):
            writer.writerow([pos[i, 0], pos[i, 1], u_exact[i], u_pred[i], abs_error[i]])

    print(f"\nExported {len(u_exact)} points to {out_file}")


if __name__ == "__main__":
    main()
