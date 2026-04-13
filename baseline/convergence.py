"""
Vanilla PINN Baseline Runner.
Executes PINN experiments across multiple grid resolutions and saves results to CSV.
Designed for easy manual comparison with G-GRN in Excel/Origin.
"""
import os
import torch
import numpy as np

from utils import set_seed
from data import MMSDataGenerator
from train import Trainer

from .pinn import VanillaPINN, PINNStrongFormLoss

def run_convergence_study(args, device):
    """Executes PINN across given resolutions and logs results."""
    res_arg = getattr(args, 'resolutions', '16,24,32,64')
    
    # 智能处理 resolutions
    if isinstance(res_arg, str):
        resolutions = [int(r) for r in res_arg.split(',')]
    elif isinstance(res_arg, int):
        resolutions = [res_arg]
    else:
        resolutions = list(res_arg)
        
    print(f"\n🚀 Starting PINN Baseline Study | Resolutions: {resolutions}")

    # 用于记录最终结果的列表
    csv_data = []

    for res in resolutions:
        print(f"\n{'='*50}\nResolution: {res}x{res} (h = {2.0/res:.4f})\n{'='*50}")
        h = 2.0 / res

        # 1. 生成数据 (PINN 不需要 GFD stencils，跳过 StencilCoefficientComputer 节省时间)
        gen = MMSDataGenerator(
            resolution=res, beta_minus=args.beta_minus, beta_plus=args.beta_plus,
        )
        # PINN 不依赖 r_connect 求导，但为了兼容统一的 loss 接口（计算跳跃损失），依然传 r_connect
        r_connect = 4.5 / res
        data = gen.build_graph(r_connect).to(device)

        # 2. 训练 Vanilla PINN
        set_seed(args.seed)
        model_pinn = VanillaPINN(hidden_channels=args.hidden_channels, num_layers=args.num_layers)
        crit_pinn = PINNStrongFormLoss(
            w_pde=args.w_pde, w_bc=args.w_bc, w_jump=args.w_jump
        )
        
        _, metrics_pinn, train_time = Trainer(model_pinn, device).fit(
            data, crit_pinn, epochs=args.epochs, lr=args.lr, 
            phase_name=f"PINN (res={res})", log_interval=1000
        )

        # 3. 收集结果 (与 GGRN 的 mms_metrics.json 保持一致)
        csv_data.append({
            'Resolution': res,
            'h': h,
            'MSE': metrics_pinn['mse'],
            'Rel_L2': metrics_pinn['rel_l2'],
            'Rel_Linf': metrics_pinn['rel_linf'],
            'Abs_L2': metrics_pinn['abs_l2'],
            'Abs_Linf': metrics_pinn['abs_linf'],
            'Max_Error': metrics_pinn['max_error'],
            'Time(s)': train_time
        })

    # ==========================================
    # 将结果保存为 CSV，方便直接复制到 Excel
    # ==========================================
    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, 'pinn_baseline_results.csv')
    
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("Resolution,h,MSE,Rel_L2,Rel_Linf,Abs_L2,Abs_Linf,Max_Error,Time(s)\n")
        for row in csv_data:
            f.write(f"{row['Resolution']},{row['h']:.6f},{row['MSE']:.6e},{row['Rel_L2']:.6e},"
                    f"{row['Rel_Linf']:.6e},{row['Abs_L2']:.6e},{row['Abs_Linf']:.6e},"
                    f"{row['Max_Error']:.6e},{row['Time(s)']:.2f}\n")
            
    print(f"\n✅ PINN Baseline Study Completed!")
    print(f"📝 Results successfully saved to: {csv_path}")
    print("👉 You can now open this CSV in Excel and compare it with your G-GRN logs.")