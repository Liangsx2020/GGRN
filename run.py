"""
Universal Entry Point for G-GRN.

Usage Examples:
    python run.py --case mms
    python run.py --case oscillating --epochs 5000
    python run.py --case elliptic --config configs/case2/config.yaml
    python run.py --case convergence --resolutions 16,32,64 --epochs 1000
"""
import os
import torch
import json

from utils import get_args, set_seed, plot_results
from data import MMSDataGenerator, OscillatingDataGenerator, EllipticInterfaceDataGenerator, StencilCoefficientComputer
from model import GGRN
from loss import ConsistentStrongFormLoss
from train import Trainer
from baseline import run_convergence_study  # <-- 引入我们刚才写的 Baseline 包


def main():
    # 1. Initialization
    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print(f"🚀 Running Case: {args.case.upper()}")
    print("=" * 60)

    # ==========================================================
    # ROUTE A: Convergence Study (Baseline Comparison)
    # ==========================================================
    if args.case == 'convergence':
        run_convergence_study(args, device)
        return  # End execution here for convergence study

    # ==========================================================
    # ROUTE B: Standard G-GRN Forward Problems
    # ==========================================================
    print(f"Device: {device} | Resolution: {args.resolution} | Epochs: {args.epochs}")
    print(f"Beta: β⁻={args.beta_minus}, β⁺={args.beta_plus}")

    # 2. Data Pipeline
    print("\n[1/4] Generating Mesh and Graph...")
    if args.case == 'mms':
        gen = MMSDataGenerator(resolution=args.resolution, beta_minus=args.beta_minus, beta_plus=args.beta_plus, data_frac=args.data_frac)
    elif args.case == 'oscillating':
        gen = OscillatingDataGenerator(
            resolution=args.resolution, beta_minus=args.beta_minus, beta_plus=args.beta_plus,
            m=getattr(args, 'm', 3), epsilon=getattr(args, 'epsilon', 0.3),
            data_frac=args.data_frac
        )
    elif args.case == 'elliptic':
        gen = EllipticInterfaceDataGenerator(
            resolution=args.resolution, beta_minus=args.beta_minus, beta_plus=args.beta_plus,
            a=getattr(args, 'a', 0.6), b=getattr(args, 'b', 0.4),
            data_frac=args.data_frac
        )
    else:
        raise ValueError(f"Unknown case: {args.case}")

    data = gen.build_graph(r_connect=args.r_connect, verbose=True)
    
    print("\n[2/4] Computing GFD Stencils...")
    data = StencilCoefficientComputer(max_order=args.max_order).compute_stencils(data)
    data = data.to(device)

    # 3. Model & Trainer
    print(f"\n[3/4] Initializing Model (hidden={args.hidden_channels}, layers={args.num_layers})...")
    model = GGRN(hidden_channels=args.hidden_channels, num_layers=args.num_layers).to(device)
    trainer = Trainer(model, device)

    # 4. Training Execution
    if args.case == 'elliptic':
        phase1_epochs = args.epochs // 3
        phase2_epochs = args.epochs - phase1_epochs
        
        crit_p1 = ConsistentStrongFormLoss(w_pde=0.0, w_bc=0.0, w_jump=0.0, w_data=1000.0)
        trainer.fit(data, crit_p1, epochs=phase1_epochs, lr=args.lr * 2, phase_name="Phase 1 (Data-Driven)")
        
        crit_p2 = ConsistentStrongFormLoss(w_pde=args.w_pde, w_bc=args.w_bc, w_jump=args.w_jump, w_data=100.0,
                                           w_j1=args.w_j1, w_j2=args.w_j2, j2_scale=args.j2_scale)
        hist, metrics, t_time = trainer.fit(data, crit_p2, epochs=phase2_epochs, lr=args.lr * 0.5, phase_name="Phase 2 (Physics-Informed)")
    else:
        criterion = ConsistentStrongFormLoss(
            w_pde=args.w_pde, w_bc=args.w_bc, w_jump=args.w_jump, w_data=args.w_data,
            w_j1=args.w_j1, w_j2=args.w_j2, j2_scale=args.j2_scale
        )
        hist, metrics, t_time = trainer.fit(
            data, criterion, epochs=args.epochs, lr=args.lr, 
            scheduler_type=args.scheduler, grad_clip=args.grad_clip, 
            log_interval=args.log_interval, phase_name="Training"
        )

    # 5. Save Results
    print(f"\n[4/4] Saving results to {args.save_dir}...")
    filename = f"{args.case}_r{args.resolution}_ep{args.epochs}.png"
    plot_results(model, data, hist, args.save_dir, filename)

    metrics['train_time'] = t_time
    
    formatted_metrics = {
        k: f"{v:.5e}" if isinstance(v, float) else v 
        for k, v in metrics.items()
    }
    
    metrics_file = os.path.join(args.save_dir, f"{args.case}_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(formatted_metrics, f, indent=4)
    print(f"📝 Metrics saved to: {metrics_file}")
    
    print("\n" + "=" * 60)
    print("🎉 ALL TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()