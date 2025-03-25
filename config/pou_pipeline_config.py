# 全局设置
seed = 42
output_dir = "./outputs/figs_pipeline"
device = "gpu"  # 或 "cpu"

# Phase A: SinglePINN 训练参数
pinn = {
    "learning_rate": 1e-3,
    "max_steps": 5000,
    "tol": 0.5,
    "num_collocation": 1000
}

# Phase B: RBF-POU 训练参数
pou = {
    "num_partitions": 2,
    "lambda_reg": 0.1,
    "lr_phase1": 0.1,
    "lr_phase2": 0.05,
    "num_epochs_phase1": 3000,
    "num_epochs_phase2": 1000
}

# Phase C: FBPINN 训练参数
fbpinn = {
    "learning_rate": 1e-3,
    "num_collocation": 1000,
    "train_steps": 30000
}
from physics.pde_cosine import pde_problem_cosine  # 或其他问题，如 pde_poisson, pde_wave 等

pde = pde_problem_cosine
