#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pipeline_06.py

三阶段流程:
Phase A -> PDE PINN (domain=[0,6])
Phase B -> RBF POU(二阶多项式), 2-phase LSGD (num_partitions=5)
Phase C -> 用(5个) POU window + FB-PINN，再解 PDE，并可视化 window function

会在脚本开始处检查JAX当前使用的设备, 若出现 [GpuDevice(id=0)] 等, 说明在用GPU.
"""

import os
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import numpy as onp

# ================================
# 全局配置
# ================================
output_dir = "./figs_pipeline_c"
os.makedirs(output_dir, exist_ok=True)

print("[INFO] JAX devices:", jax.devices())  # 检查 GPU/CPU

# -------------------------------
# PDE 定义 (domain=[0,6])
# -------------------------------
def phi(x):
    """
    phi(x)= (pi/4)* x^2
    在 x=6 时 => phi(6)= (pi/4)*36= 9*pi => sin(9*pi)=0 => 满足边界u(6)=0
    """
    return (jnp.pi / 4.0)* (x**2)

def u_exact(x):
    """
    PDE 真解: u(x)= sin((pi/4)* x^2).
    在[0,6]内, u(0)=0, u(6)=sin(9*pi)=0
    """
    return jnp.sin(phi(x))

def f_pde(x):
    """
    PDE: u''(x)+ f(x)=0
    若 u= sin((pi/4)* x^2), => f(x)= -u''(x).
    参考:
      u'(x)= cos(phi(x))*phi'(x)= cos(...)*(pi/2 * x)
      u''(x)= d/dx [ (pi/2)* x cos(...) ]= ...
    最终 => f(x)= (pi^2/4)*x^2 sin(...) - (pi/2)* cos(...)
    """
    return (jnp.pi**2 / 4.0)* x**2 * jnp.sin(phi(x)) - (jnp.pi/2.0)* jnp.cos(phi(x))

# ================================
# Phase A: SinglePINN
# ================================
class SinglePINN(eqx.Module):
    mlp: eqx.nn.MLP
    def __init__(self, key):
        self.mlp = eqx.nn.MLP(
            in_size=1, out_size=1, width_size=20, depth=3,
            activation=jax.nn.tanh, key=key
        )
    def __call__(self, x):
        x_arr = jnp.array([x])
        return self.mlp(x_arr)[0]

def pde_residual_single(params, x):
    """
    计算: u''(x)+ f_pde(x)
    """
    def u_fun(xx):
        return params(xx)
    dudx = jax.grad(u_fun)
    d2udx2 = jax.grad(dudx)
    return d2udx2(x) + f_pde(x)

def loss_single_pinn(params, x_colloc):
    # PDE残差
    res_vals = jax.vmap(lambda xx: pde_residual_single(params, xx))(x_colloc)
    loss_pde = jnp.mean(res_vals**2)

    # Dirichlet BC: u(0)=0, u(6)=0
    bc_left = params(0.0)
    bc_right = params(6.0)
    loss_bc = bc_left**2 + bc_right**2
    return loss_pde + loss_bc

@eqx.filter_jit
def train_step_single(params, opt_state, x_c, optimizer):
    val, grads = eqx.filter_value_and_grad(loss_single_pinn)(params, x_c)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, val

def train_single_pinn_pde(tol=0.1, max_steps=100000):
    """
    在[0,6]上解 PDE: u''+f=0, BC: u(0)=0, u(6)=0
    """
    key = jr.PRNGKey(42)
    model = SinglePINN(key)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # collocation 点在 [0,6]
    x_colloc = jr.uniform(jr.PRNGKey(999), shape=(1000,), minval=0, maxval=6)

    for it in range(max_steps):
        model, opt_state, loss_val = train_step_single(model, opt_state, x_colloc, optimizer)
        # 每 1000 步检查一次
        if it % 1000 == 0:
            x_eval = jnp.linspace(0,6,300)
            u_pred = jax.vmap(model)(x_eval)
            u_ref = u_exact(x_eval)
            l1_err = jnp.mean(jnp.abs(u_pred - u_ref))
            print(f"[SinglePINN] it={it}, loss={loss_val:.4e}, L1={l1_err:.4f}")
            if l1_err < tol:
                print(f"[SinglePINN] L1={l1_err:.4f} < {tol}, stop.")
                break
    return model

# ================================
# Phase B: RBF POU + 2-phase LSGD(二阶多项式)
# ================================
def init_rbf_params(rng_key, num_centers):
    key1, key2 = jax.random.split(rng_key)
    centers = jax.random.normal(key1, shape=(num_centers,))
    widths = jnp.ones((num_centers,)) * 0.2
    return {"centers": centers, "widths": widths}

def rbf_forward(params, x):
    """
    x shape=[N], => partitions shape=[N, C]
    """
    centers = params["centers"]
    widths = params["widths"]
    dist_sq = (x[:,None] - centers[None,:])**2
    raw = jnp.exp(- dist_sq / (widths**2))
    sums = jnp.sum(raw, axis=1, keepdims=True) + 1e-12
    return raw / sums

def fit_local_polynomials_2nd_order(x_np, y_np, part_np):
    """
    每个子域: 二阶多项式加权拟合
    """
    N, C = part_np.shape
    coeffs = []
    for i in range(C):
        w_i = part_np[:, i]
        W = onp.diag(w_i)
        X = onp.vstack([onp.ones_like(x_np), x_np, x_np**2]).T  # (N,3)
        c = onp.linalg.lstsq(W@X, W@y_np, rcond=None)[0]  # shape=[3,]
        coeffs.append(c)
    return onp.array(coeffs)

def compute_loss_pou(params, x_j, y_j, num_partitions, lambda_reg):
    partitions = rbf_forward(params, x_j)  # shape=[N,C]
    # 停梯度后转 numpy
    part_np = onp.array(jax.lax.stop_gradient(partitions))
    x_np = onp.array(jax.lax.stop_gradient(x_j))
    y_np = onp.array(jax.lax.stop_gradient(y_j))

    # 二阶多项式拟合
    local_coeffs = fit_local_polynomials_2nd_order(x_np, y_np, part_np)
    c0, c1, c2 = local_coeffs[:,0], local_coeffs[:,1], local_coeffs[:,2]

    # 合成逼近
    y_pou = jnp.sum(
        partitions * (c0[None,:] + c1[None,:]* x_j[:,None] + c2[None,:]*(x_j[:,None]**2)),
        axis=1
    )
    # MSE
    mse = jnp.mean((y_pou - y_j)**2)
    # L2正则
    reg_val = jnp.linalg.norm(partitions, ord=2)
    return mse + lambda_reg* reg_val

loss_and_grad_pou = jax.value_and_grad(compute_loss_pou)

def train_two_phase_lsgd(params_init, x_data, y_data, num_partitions=5,
                         num_epochs_phase1=10000, num_epochs_phase2=10000,
                         lambda_reg=0.1, lr_phase1=0.1, lr_phase2=0.05):
    """
    2-phase 训练: RBF + 2nd poly, num_partitions=5
    """
    opt1 = optax.sgd(lr_phase1)
    opt2 = optax.sgd(lr_phase2)

    opt_state1 = opt1.init(params_init)
    opt_state2 = opt2.init(params_init)

    params_cur = params_init
    x_j = jnp.array(x_data)
    y_j = jnp.array(y_data)

    # Phase1
    for ep in range(num_epochs_phase1):
        lv, grads = loss_and_grad_pou(params_cur, x_j, y_j, num_partitions, lambda_reg)
        updates, opt_state1 = opt1.update(grads, opt_state1, params_cur)
        params_cur = optax.apply_updates(params_cur, updates)
        if ep % 200 == 0:
            print(f"[POU Phase1] ep={ep}, loss={lv:.6f}")
    params_phase1 = params_cur

    # Phase2
    for ep in range(num_epochs_phase2):
        lv, grads = loss_and_grad_pou(params_cur, x_j, y_j, num_partitions, 0.0)
        updates, opt_state2 = opt2.update(grads, opt_state2, params_cur)
        params_cur = optax.apply_updates(params_cur, updates)
        if ep % 200 == 0:
            print(f"[POU Phase2] ep={ep}, loss={lv:.6f}")
    params_phase2 = params_cur
    return params_phase1, params_phase2

def fit_and_infer(params, x_data, y_data):
    """
    用 RBF + 二阶多项式 => 全局逼近
    """
    x_j = jnp.array(x_data)
    partitions = rbf_forward(params, x_j)
    part_np = onp.array(jax.lax.stop_gradient(partitions))
    coeffs = fit_local_polynomials_2nd_order(
        onp.array(x_data), onp.array(y_data), part_np
    )
    c0, c1, c2 = coeffs[:,0], coeffs[:,1], coeffs[:,2]

    y_pou = jnp.sum(
        partitions*(c0[None,:] + c1[None,:]* x_j[:,None] + c2[None,:]*(x_j[:,None]**2)),
        axis=1
    )
    return onp.array(y_pou)

def visualize_pou(params, x_data, fig_name):
    x_j = jnp.array(x_data)
    p = rbf_forward(params, x_j)
    p_np = onp.array(p)
    plt.figure(figsize=(8,4))
    for i in range(p_np.shape[1]):
        plt.plot(x_data, p_np[:,i], label=f"Partition {i+1}")
    plt.xlabel("x"); plt.ylabel("Weight")
    plt.title(fig_name)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, fig_name+".png"), dpi=300)
    plt.close()

# ================================
# Phase C: FBPINN + 5-Window
# ================================
class FBPINN(eqx.Module):
    subnets: tuple
    def __init__(self, n_sub, key):
        keys = jr.split(key, n_sub)
        self.subnets= tuple(
            eqx.nn.MLP(
                in_size=1, out_size=1, width_size=20, depth=3,
                activation=jax.nn.tanh, key=keys[i]
            )
            for i in range(n_sub)
        )

def net_eval(net, x):
    x_arr = jnp.array([x])
    return net(x_arr)[0]

def fbpinn_global_sol(x, model, partition_func):
    w = partition_func(x) # shape=[C]
    val=0.0
    for i, net_i in enumerate(model.subnets):
        val += w[i]* net_eval(net_i, x)
    return val

def global_residual(x, model, partition_func):
    """
    PDE: u''+ f=0, 在 domain=[0,6]
    """
    def u_g(xx):
        return fbpinn_global_sol(xx, model, partition_func)
    dudx = jax.grad(u_g)
    d2udx2 = jax.grad(dudx)
    return d2udx2(x) + f_pde(x)

def loss_fbpinn(model, x_colloc, partition_func):
    # PDE 残差
    r_vals = jax.vmap(lambda xx: global_residual(xx, model, partition_func))(x_colloc)
    loss_pde = jnp.mean(r_vals**2)
    # 边界: u(0)=0, u(6)=0
    bc_left = fbpinn_global_sol(0.0, model, partition_func)
    bc_right = fbpinn_global_sol(6.0, model, partition_func)
    bc_loss = bc_left**2 + bc_right**2
    return loss_pde + bc_loss

@eqx.filter_jit
def train_step_fbpinn(params, opt_state, x_c, partition_func, optimizer):
    lv, grads = eqx.filter_value_and_grad(loss_fbpinn)(params, x_c, partition_func)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, lv

def train_fbpinn_with_window(window_func, n_sub=5):
    key = jr.PRNGKey(2023)
    model = FBPINN(n_sub, key)

    # 全局 collocation, domain=[0,6]
    x_colloc = jr.uniform(jr.PRNGKey(1234), shape=(1000,), minval=0, maxval=6)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    loss_hist=[]
    for it in range(30000):
        model, opt_state, lv = train_step_fbpinn(model, opt_state, x_colloc, window_func, optimizer)
        if it%2000==0:
            print(f"[FBPINN] it={it}, loss={lv:.6f}")
        loss_hist.append(lv)
    return model, loss_hist


# ================================
# 主流程
# ================================
def main():
    # 显示 JAX 设备 (检查GPU)
    print("[INFO] JAX devices:", jax.devices())

    # --- Phase A ---
    print("=== Phase A: SinglePINN (domain=[0,6]) ===")
    pinn_model = train_single_pinn_pde(tol=0.5, max_steps=50000)

    x_plot = jnp.linspace(0,6,300)
    u_pin = jax.vmap(pinn_model)(x_plot)
    u_exa = u_exact(x_plot)
    err_l1_pin = jnp.mean(jnp.abs(u_pin - u_exa))
    print(f"[SinglePINN] L1={err_l1_pin:.4f}")

    plt.figure()
    plt.plot(x_plot, u_exa, "--", label="Exact PDE on [0,6]")
    plt.plot(x_plot, u_pin, label=f"PINN (L1={err_l1_pin:.4f})")
    plt.title("Phase A: SinglePINN PDE (domain=[0,6])")
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(output_dir, "phaseA_singlepinn.png"), dpi=300)
    plt.close()

    # --- Phase B ---
    print("\n=== Phase B: POU 2-phase (5 subdomains) ===")
    # 采样 (x,y)
    N=500
    x_data = onp.linspace(0,6,N)
    y_data = onp.array([ float(pinn_model(jnp.array(xx))) for xx in x_data ])

    rng_pou = jr.PRNGKey(555)
    num_sub=5  # 5 subdomains
    params_init = init_rbf_params(rng_pou, num_sub)

    # 2-phase
    p1, p2 = train_two_phase_lsgd(
        params_init, x_data, y_data,
        num_partitions=num_sub,
        num_epochs_phase1=30000,
        num_epochs_phase2=10000,
        lambda_reg=0.1,
        lr_phase1=0.1,
        lr_phase2=0.05
    )

    # Phase1
    y_p1 = fit_and_infer(p1, x_data, y_data)
    plt.figure()
    plt.plot(x_data, y_data, '--', label="PINN-based Data")
    plt.plot(x_data, y_p1, label="POU(2nd) - Phase1")
    plt.title("Phase B: POU(2nd) Phase1")
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(output_dir, "phaseB_pou_phase1.png"), dpi=300)
    plt.close()

    visualize_pou(p1, x_data, "phaseB_pou_partition_phase1")

    # Phase2
    y_p2 = fit_and_infer(p2, x_data, y_data)
    plt.figure()
    plt.plot(x_data, y_data, '--', label="PINN-based Data")
    plt.plot(x_data, y_p2, label="POU(2nd) - Phase2")
    plt.title("Phase B: POU(2nd) Phase2")
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(output_dir, "phaseB_pou_phase2.png"), dpi=300)
    plt.close()

    visualize_pou(p2, x_data, "phaseB_pou_partition_phase2")

    # --- Phase C ---
    print("\n=== Phase C: FBPINN using POU window (5 subdomains) on [0,6] ===")

    def partition_func(x):
        # x可能是标量或者数组
        x_arr = jnp.atleast_1d(x)
        part_ = rbf_forward(p2, x_arr)
        if x_arr.shape==(1,):
            return part_[0]
        return part_

    # 先可视化 phaseC 的 window
    x_pouC = jnp.linspace(0,6,300)
    w_pouC = partition_func(x_pouC)  # shape=(300,5)
    w_pouC_np = onp.array(w_pouC)
    plt.figure()
    for i in range(w_pouC_np.shape[1]):
        plt.plot(x_pouC, w_pouC_np[:,i], label=f"Window {i+1}")
    plt.title("Phase C: FBPINN POU Window (5 subdomains) [0,6]")
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(output_dir, "phaseC_window_5sub.png"), dpi=300)
    plt.close()

    # 训练FBPINN
    fbpinn_model, loss_hist = train_fbpinn_with_window(partition_func, n_sub=num_sub)

    plt.figure()
    plt.plot(loss_hist, label="FBPINN Loss")
    plt.yscale("log")
    plt.title("Phase C: FBPINN Loss")
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(output_dir,"phaseC_fbpinn_loss.png"), dpi=300)
    plt.close()

    # 评估
    u_fbp = jax.vmap(lambda xx: fbpinn_global_sol(xx, fbpinn_model, partition_func))(x_plot)
    err_l1_fbp = jnp.mean(jnp.abs(u_fbp - u_exa))
    print(f"[FBPINN] L1 final= {err_l1_fbp:.4f}")

    plt.figure()
    plt.plot(x_plot, u_exa, "--", label="Exact PDE (domain=[0,6])")
    plt.plot(x_plot, u_fbp, label=f"FBPINN(POU) L1={err_l1_fbp:.4f}")
    plt.title("Phase C: FBPINN Result (5 subdomains window) [0,6]")
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(output_dir,"phaseC_fbpinn_result.png"), dpi=300)
    plt.close()

    print("Done. Figures saved in", output_dir)

if __name__=="__main__":
    main()
