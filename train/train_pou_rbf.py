# train/train_pou_rbf.py
import jax
import jax.numpy as jnp
import optax
import numpy as onp
from model.rbf_pou import rbf_forward, fit_local_polynomials_2nd_order

def compute_loss_pou(params, x, y, num_partitions, lambda_reg):
    """
    Loss = MSE(全局逼近, y) + lambda_reg * L2范数(分区矩阵)
    """
    partitions = rbf_forward(params, x)  # shape (N, C)
    # 将分区矩阵停梯度，转为 numpy 计算局部二阶多项式拟合系数
    partitions_np = onp.array(jax.lax.stop_gradient(partitions))
    x_np = onp.array(jax.lax.stop_gradient(x))
    y_np = onp.array(jax.lax.stop_gradient(y))
    coeffs = fit_local_polynomials_2nd_order(x_np, y_np, partitions_np)
    c0, c1, c2 = coeffs[:,0], coeffs[:,1], coeffs[:,2]
    y_pou = jnp.sum(
        partitions * (jnp.array(c0)[None, :] + jnp.array(c1)[None, :]* x[:, None] + jnp.array(c2)[None, :]*(x[:, None]**2)),
        axis=1
    )
    mse = jnp.mean((y_pou - y)**2)
    reg_val = jnp.linalg.norm(partitions, ord=2)
    return mse + lambda_reg * reg_val

compute_loss_and_grad_pou = jax.value_and_grad(compute_loss_pou)

def train_pou_rbf(params_init, x_data, y_data, num_partitions, lambda_reg, lr_phase1, lr_phase2, num_epochs_phase1, num_epochs_phase2):
    """
    Two-phase 训练 RBF-POU 参数
    """
    opt1 = optax.sgd(lr_phase1)
    opt2 = optax.sgd(lr_phase2)
    opt_state1 = opt1.init(params_init)
    opt_state2 = opt2.init(params_init)
    params_cur = params_init
    x = x_data
    y = y_data

    # Phase 1
    for ep in range(num_epochs_phase1):
        loss_val, grads = compute_loss_and_grad_pou(params_cur, x, y, num_partitions, lambda_reg)
        updates, opt_state1 = opt1.update(grads, opt_state1, params_cur)
        params_cur = optax.apply_updates(params_cur, updates)
        if ep % 200 == 0:
            print(f"[POU Phase1] Epoch {ep}, Loss: {loss_val:.6f}")
    params_phase1 = params_cur

    # Phase 2 (without regularization: lambda=0)
    for ep in range(num_epochs_phase2):
        loss_val, grads = compute_loss_and_grad_pou(params_cur, x, y, num_partitions, 0.0)
        updates, opt_state2 = opt2.update(grads, opt_state2, params_cur)
        params_cur = optax.apply_updates(params_cur, updates)
        if ep % 200 == 0:
            print(f"[POU Phase2] Epoch {ep}, Loss: {loss_val:.6f}")
    params_phase2 = params_cur
    return params_phase1, params_phase2
