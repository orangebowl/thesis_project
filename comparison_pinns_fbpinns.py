#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import matplotlib.pyplot as plt

# ============ 全局设置 ============
output_dir = "./figs_compare_fixed_points_finalL1"
os.makedirs(output_dir, exist_ok=True)

domain_min, domain_max = 0.0, 8.0
n_steps = 200000         # 训练步数
learning_rate = 1e-3    # 学习率
sigma_val = 0.17        # FB-PINN 的窗口函数平滑度
n_collocation = 1000    # 让两种方法共享的 collocation points 总数

print("[INFO] JAX devices:", jax.devices())

# ========== 1) PDE 定义 ==========
def phi(x):
    return (jnp.pi / 4.0) * (x**2)

def u_exact(x):
    return jnp.sin(phi(x))

def f_pde(x):
    return (jnp.pi**2 / 4.0) * x**2 * jnp.sin(phi(x)) - (jnp.pi / 2.0) * jnp.cos(phi(x))

# ========== 2) Ansatz (自动满足边界) ==========
def ansatz(x, net_out):
    """
    A(x) = (1 - e^{-x})(1 - e^{-(8 - x)})
    满足 u(0)=0, u(8)=0
    """
    return (1.0 - jnp.exp(-x)) * (1.0 - jnp.exp(-(8.0 - x))) * net_out

# ========== 3) 全局采样点 (共享) ==========
sampling_key = jr.PRNGKey(999)
x_collocation = jr.uniform(sampling_key, (n_collocation,),
                           minval=domain_min, maxval=domain_max)

# ========== 4) Vanilla PINN ==========
class SinglePINN(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key):
        self.mlp = eqx.nn.MLP(
            in_size=1,
            out_size=1,
            width_size=20,
            depth=3,
            activation=jax.nn.tanh,
            key=key
        )

    def __call__(self, x):
        # PINN ouput = ansatz(x, mlp(x))
        x_in = jnp.atleast_1d(x)
        net_out = self.mlp(x_in.reshape((1,)))  # shape (1,1)
        return ansatz(x, net_out[0])

def pde_residual(model, x):
    """PDE 残差: u''(x) + f_pde(x)"""
    def u_x(z):
        return model(z)
    d2udx2 = jax.grad(jax.grad(u_x))(x)
    return d2udx2 + f_pde(x)

def loss_single_pinn(params):
    """
    单域 PINN 只计算 PDE 残差平方的平均
    """
    residuals = jax.vmap(lambda xx: pde_residual(params, xx))(x_collocation)
    return jnp.mean(residuals**2)

@eqx.filter_jit
def train_step_single(params, opt_state, optimizer):
    lv, grads = eqx.filter_value_and_grad(loss_single_pinn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, lv

def train_single_pinn():
    """
    训练并返回:
    - 模型
    - 每次迭代的训练损失列表 (不记录中间 L1)
    """
    key = jr.PRNGKey(42)
    model = SinglePINN(key)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    train_loss_list = []

    for i in range(n_steps):
        model, opt_state, lv = train_step_single(model, opt_state, optimizer)
        train_loss_list.append(lv)

        if i % 1000 == 0:
            print(f"[PINN] Step={i}, TrainLoss={lv:.3e}")

    return model, train_loss_list

# ========== 5) FB-PINN (4 子域) + Sigmoid 窗口 ==========
def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))

def sigmoid_window_function(x, subdomains, sigma):
    x = jnp.atleast_1d(x)
    subdomains = jnp.array(subdomains)

    left = sigmoid((x[:, None] - subdomains[:, 0]) / sigma)
    right = sigmoid((subdomains[:, 1] - x[:, None]) / sigma)
    w = left * right
    sum_w = jnp.sum(w, axis=1, keepdims=True) + 1e-10
    return w / sum_w

class SmoothFBPINN(eqx.Module):
    subnets: tuple
    subdomains: jnp.ndarray
    sigma: float

    def __init__(self, subdomains, sigma, key):
        self.subdomains = jnp.array(subdomains)
        self.sigma = sigma

        n_sub = len(subdomains)
        keys = jr.split(key, n_sub)
        self.subnets = tuple(
            eqx.nn.MLP(
                in_size=1,
                out_size=1,
                width_size=20,
                depth=3,
                activation=jax.nn.tanh,
                key=keys[i]
            )
            for i in range(n_sub)
        )

    def __call__(self, x):
        x_in = jnp.atleast_1d(x)
        w = sigmoid_window_function(x_in, self.subdomains, self.sigma)
        subvals = jnp.array([net(x_in.reshape((1,)))[0] for net in self.subnets])
        net_sum = jnp.dot(w[0, :], subvals)
        return ansatz(x, net_sum)

def loss_fbpinn(params, x_subsets):
    total_res = 0.0
    count = 0
    for x_local in x_subsets:
        if len(x_local) == 0:
            continue
        res = jax.vmap(lambda xx: pde_residual(params, xx))(x_local)
        total_res += jnp.sum(res**2)
        count += len(x_local)
    return total_res / (count + 1e-10)

@eqx.filter_jit
def train_step_fbpinn(params, opt_state, x_subsets, optimizer):
    lv, grads = eqx.filter_value_and_grad(loss_fbpinn)(params, x_subsets)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, lv

def train_smooth_fbpinn():
    """
    均分成 4 个子域: [0,2],[2,4],[4,6],[6,8]
    在 x_collocation 中按照子域范围划分
    """
    num_sub = 4
    subdomains = [(i*2.0, (i+1)*2.0) for i in range(num_sub)]

    x_subsets = [x_collocation[(x_collocation >= a) & (x_collocation < b)]
                 for (a, b) in subdomains]

    key = jr.PRNGKey(2022)
    model = SmoothFBPINN(subdomains, sigma_val, key)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    train_loss_list = []

    for it in range(n_steps):
        model, opt_state, lv = train_step_fbpinn(model, opt_state, x_subsets, optimizer)
        train_loss_list.append(lv)

        if it % 1000 == 0:
            print(f"[FBPINN] Step={it}, TrainLoss={lv:.3e}")

    return model, train_loss_list, subdomains

# ========== 绘图函数 ==========
def plot_window_functions(x, subdomains, sigma, fig_name):
    """绘制 4 个子域的 Sigmoid 窗口函数"""
    w = sigmoid_window_function(x, jnp.array(subdomains), sigma)
    plt.figure()
    for i in range(w.shape[1]):
        plt.plot(x, w[:, i], label=f"Window {i+1}")
    plt.xlabel("x")
    plt.ylabel("Weight")
    plt.title("FBPINN Window Functions (4 Subdomains)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, fig_name), dpi=300)
    plt.close()

# ========== 主函数 ==========
def main():

    # --- 训练 FB-PINN ---
    print("Training FB-PINN...")
    fb_model, fb_loss, subdomains = train_smooth_fbpinn()
    
    # --- 训练 Single PINN ---
    print("Training Single PINN...")
    pinn_model, pinn_loss = train_single_pinn()

    # --- 绘制 Training Loss 对比 ---
    plt.figure()
    plt.plot(fb_loss, label="FBPINN (4 subdomains)")
    plt.plot(pinn_loss, label="Single PINN")
    plt.yscale("log")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "train_loss_compare.png"), dpi=300)
    plt.close()

    # --- (4) 仅在训练结束后计算最终 L1 误差 ---
    x_eval = jnp.linspace(domain_min, domain_max, 300)
    pinn_pred = jax.vmap(pinn_model)(x_eval)
    fb_pred = jax.vmap(fb_model)(x_eval)
    exact_sol = u_exact(x_eval)

    l1_pinn_final = jnp.mean(jnp.abs(pinn_pred - exact_sol))
    l1_fb_final = jnp.mean(jnp.abs(fb_pred - exact_sol))

    print(f"\n[RESULT] Final L1 Error => PINN = {l1_pinn_final:.4e}, FB-PINN = {l1_fb_final:.4e}")

    # --- (5) 绘制窗口函数 (FB-PINN) ---
    x_plot = jnp.linspace(domain_min, domain_max, 300)
    plot_window_functions(x_plot, subdomains, sigma_val, "window_weights_fbPINN_4sub.png")

    # --- (6) 最终解对比 ---
    plt.figure()
    
    plt.plot(x_eval, pinn_pred, label=f"Single PINN (L1={l1_pinn_final:.4e})")
    plt.plot(x_eval, fb_pred,   label=f"FBPINN-4  (L1={l1_fb_final:.4e})")
    plt.plot(x_eval, exact_sol, "--", label="Exact Solution")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("Solution Comparison (Final L1 Error)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "solution_compare.png"), dpi=300)
    plt.close()

    print("\n==== 训练完成，结果保存在:", output_dir)

if __name__ == "__main__":
    main()
