from config import *
from model.pinn_model import PINN
from model.fbpinn_model import SmoothFBPINN
from train.trainer_single import train_single
from train.trainer_fbpinn import train_fbpinn
from utils.visualizer import plot_loss_compare, plot_solution_compare, plot_window_functions
from physics.pde_1d import u_exact
import jax
import jax.random as jr
import jax.numpy as jnp
import os

os.makedirs("outputs/figures", exist_ok=True)

# ---- 采样点
sampling_key = jr.PRNGKey(0)
x_collocation = jr.uniform(sampling_key, (n_collocation,), minval=domain_min, maxval=domain_max)

# ---- 分割子域
subdomains = [(i * 2.0, (i + 1) * 2.0) for i in range(n_subdomains)]
x_subsets = [x_collocation[(x_collocation >= a) & (x_collocation < b)] for a, b in subdomains]

# ---- 训练 FBPINN
fb_model = SmoothFBPINN(subdomains, sigma_val, jr.PRNGKey(2022))
fb_model, fb_loss = train_fbpinn(fb_model, x_subsets, n_steps, learning_rate)

# ---- 训练 Single PINN
single_model = PINN(jr.PRNGKey(42))
single_model, single_loss = train_single(single_model, x_collocation, learning_rate, n_steps)

# ---- 可视化 Loss 对比
plot_loss_compare(single_loss, fb_loss, "outputs/figures/loss_compare.png")

# ---- 计算预测误差
x_eval = jnp.linspace(domain_min, domain_max, 300)
fb_pred = jax.vmap(fb_model)(x_eval)
pinn_pred = jax.vmap(single_model)(x_eval)
u_true = u_exact(x_eval)

# ---- 可视化解对比 + 窗口函数
plot_solution_compare(x_eval, pinn_pred, fb_pred, u_true, "outputs/figures/solution_compare.png")
plot_window_functions(x_eval, subdomains, sigma_val, "outputs/figures/window_functions.png")
