import os

# ============================================================================
# Batch Experiment Runner for G-GRN
# ============================================================================

RESOLUTIONS = [16, 24, 32, 64]
NUM_LAYERS = 2

# --- Case 1: MMS Circular Interface ---
# Already completed. Uncomment to re-run.
# for beta_m, beta_p in [(1, 10), (1, 100), (10, 1)]:
#     for res in RESOLUTIONS:
#         save_dir = f"./results/case1/beta-m-p-{beta_m}-{beta_p}-r-{res}--5%"
#         os.system(
#             f"python run.py --case mms --config configs/case1/beta-m-p-{beta_m}-{beta_p}.yaml "
#             f"--resolution {res} --num_layers {NUM_LAYERS} --epochs 1000 "
#             f"--save_dir {save_dir}"
#         )

# --- Case 2: Oscillating Angular Solution ---
# for res in RESOLUTIONS:
#     save_dir = f"./results/case2/r-{res}--5%"
#     os.system(
#         f"python run.py --case oscillating --config configs/case2/config.yaml "
#         f"--resolution {res} --num_layers {NUM_LAYERS} --epochs 2000 "
#         f"--save_dir {save_dir}"
#     )

# --- Case 3: Elliptic Interface ---
for res in RESOLUTIONS:
    save_dir = f"./results/case3/r-{res}-single-phase--5%"
    os.system(
        f"python run.py --case elliptic --config configs/case3/config.yaml "
        f"--resolution {res} --num_layers {NUM_LAYERS} --epochs 2000 "
        f"--save_dir {save_dir}"
    )

# --- Convergence Study (GGRN vs PINN) ---
# Already completed. Uncomment to re-run.
# for beta_m, beta_p in [(1, 10), (1, 100), (10, 1)]:
#     os.system(
#         f"python run.py --case convergence --config configs/convergence/config.yaml "
#         f"--beta_minus {beta_m} --beta_plus {beta_p} "
#         f"--save_dir ./results/convergence/beta-m-p-{beta_m}-{beta_p}"
#     )
