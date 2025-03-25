# model/rbf_pou.py
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as onp

def init_rbf_params(rng_key, num_centers):
    key1, _ = jr.split(rng_key)
    centers = jr.normal(key1, shape=(num_centers,))
    widths = jnp.ones((num_centers,)) * 0.2
    return {"centers": centers, "widths": widths}

def rbf_forward(params, x):
    """
    Args:
        x: shape (N,)
    Returns:
        partitions: shape (N, num_centers)
    """
    centers = params["centers"]
    widths = params["widths"]
    dist_sq = (x[:, None] - centers[None, :])**2
    raw = jnp.exp(- dist_sq / (widths**2))
    sums = jnp.sum(raw, axis=1, keepdims=True) + 1e-12
    return raw / sums

def fit_local_polynomials_2nd_order(x_np, y_np, part_np):
    """
    针对每个子域，用二阶多项式拟合 (y ~ c0 + c1*x + c2*x^2)
    Args:
        x_np: shape (N,)
        y_np: shape (N,)
        part_np: shape (N, C)
    Returns:
        coeffs: shape (C, 3)
    """
    N, C = part_np.shape
    coeffs = []
    for i in range(C):
        w_i = part_np[:, i]
        # 构造加权最小二乘问题
        X = onp.vstack([onp.ones_like(x_np), x_np, x_np**2]).T  # (N,3)
        W = onp.diag(w_i)
        A = W @ X
        b = W @ y_np
        # 求解最小二乘问题
        c, _, _, _ = onp.linalg.lstsq(A, b, rcond=None)
        coeffs.append(c)
    return onp.array(coeffs)

def compute_pou_approx(params, x, y):
    """
    利用当前 RBF 分区拟合全局逼近函数。
    Args:
        params: RBF参数
        x: 输入, shape (N,)
        y: 目标, shape (N,)
    Returns:
        y_pou: 全局逼近结果, shape (N,)
    """
    partitions = rbf_forward(params, x)  # (N,C)
    # 停梯度后转为 numpy 数组进行二阶多项式拟合
    part_np = onp.array(jax.lax.stop_gradient(partitions))
    x_np = onp.array(jax.lax.stop_gradient(x))
    y_np = onp.array(jax.lax.stop_gradient(y))
    coeffs = fit_local_polynomials_2nd_order(x_np, y_np, part_np)
    c0, c1, c2 = coeffs[:,0], coeffs[:,1], coeffs[:,2]
    # 全局逼近：y_pou = sum_i partition_i * (c0_i + c1_i*x + c2_i*x^2)
    y_pou = jnp.sum(
        partitions * (jnp.array(c0)[None, :] + jnp.array(c1)[None, :]* x[:, None] + jnp.array(c2)[None, :]*(x[:, None]**2)),
        axis=1
    )
    return y_pou
