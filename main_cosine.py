from config import pou_pipeline_config as config
from dataloader.dataloader_1d import CollocationData1D

from model.pinn_model import PINN
from train.trainer_single import train_single
from train.trainer_fbpinn import train_fbpinn

from utils.visualizer import (
    plot_loss_compare,
    plot_solution_compare,
    plot_window_functions
)

from physics.pde_cosine import pde_residual_cosine as pde
from model.rbf_pou import rbf_forward, init_rbf_params
from train.train_pou_rbf import train_pou_rbf

import jax
import jax.random as jr
import jax.numpy as jnp
import numpy as onp
import os

# --------------------------
# 输出目录准备
# --------------------------
os.makedirs("outputs/figures", exist_ok=True)

# --------------------------
# 加载配置 & PDE 定义
# --------------------------
domain_min, domain_max = pde["domain"]
f_pde = pde["f_pde"]
u_exact = pde["u_exact"]
ansatz = pde["ansatz"]

learning_rate = config.fbpinn["learning_rate"]
n_steps = config.fbpinn["train_steps"]
num_subdomains = 5  # 可调
sigma_val = 0.17
pde_problem = config.pde

# --------------------------
# 数据加载器
# --------------------------
pinn_loader = CollocationData1D((domain_min, domain_max), num_points=200, seed=0)
x_collocation_pinn = pinn_loader.sample_uniform()

# --------------------------
# Phase A: PINN训练
# --------------------------
print("Training Single PINN ...")
single_model = PINN(jr.PRNGKey(42))
single_model, single_loss = train_single(
    model=single_model,
    x_collocation=x_collocation_pinn,
    lr=learning_rate,
    steps=n_steps,
    pde_residual=pde_residual_cosine
)

# --------------------------
# Phase B: 用 PINN 解构造样本训练 RBF POU
# --------------------------
x_data = onp.linspace(domain_min, domain_max, 300)
y_data = onp.array([float(single_model(jnp.array(xx))) for xx in x_data])

rng_pou = jr.PRNGKey(2024)
params_init = init_rbf_params(rng_pou, num_subdomains)

p1, p2 = train_pou_rbf(
    params_init, x_data, y_data,
    num_partitions=num_subdomains,
    lambda_reg=0.1,
    lr_phase1=0.1,
    lr_phase2=0.05,
    num_epochs_phase1=1000,
    num_epochs_phase2=500
)

# --------------------------
# 定义 window_fn：基于 RBF POU 输出
# --------------------------
def window_fn(x):
    x_arr = jnp.atleast_1d(x)
    w = rbf_forward(p2, x_arr)
    if x_arr.shape == (1,):
        return w[0]
    return w

# --------------------------
# Phase C: 使用自定义 window_fn 训练 FBPINN
# --------------------------
print("🧩 Training FBPINN using your interface ...")
fb_model, fb_loss = train_fbpinn(
    pde_problem=pde,
    window_fn=window_fn,
    n_sub=num_subdomains,
    steps=n_steps,
    lr=learning_rate
)

# --------------------------
# 可视化 loss 对比
# --------------------------
plot_loss_compare(single_loss, fb_loss, "outputs/figures/loss_compare.png")

# --------------------------
# 可视化解对比
# --------------------------
x_eval = jnp.linspace(domain_min, domain_max, 300)
pinn_pred = jax.vmap(single_model)(x_eval)
fb_pred = jax.vmap(fb_model)(x_eval)
u_true = u_exact(x_eval)

plot_solution_compare(
    x_eval, pinn_pred, fb_pred, u_true,
    "outputs/figures/solution_compare.png"
)

# --------------------------
# 可视化窗口函数
# --------------------------
from utils.visualizer import plot_window_from_fn
plot_window_from_fn(window_fn, x_eval, "outputs/figures/window_functions.png")

print("✅ 全部完成，图像保存在 outputs/figures/")
