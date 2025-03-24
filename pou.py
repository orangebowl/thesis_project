##########################
# rbf_pou_lsgd_jax_2nd_order.py
##########################
import numpy as onp  # "ordinary" NumPy, 用于最小二乘
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

##############################
# 1) 三角波 (p=2)
##############################
def triangle_wave(x, p=2):
    """
    与之前一致:
    triangle_wave(x) = 2 * |p*x - floor(p*x + 0.5)|
    """
    return 2.0 * jnp.abs(p*x - jnp.floor(p*x + 0.5))

##############################
# 2) 二阶多项式拟合
##############################
def fit_local_polynomials_2nd_order(x_np, y_np, partitions_np):
    """
    x_np, y_np, partitions_np 都是普通 NumPy array,
    形状分别为 [N], [N], [N, num_partitions].
    
    我们做二阶多项式(1, x, x^2)加权最小二乘:
      c = (c0, c1, c2).
    
    返回 coefficients 形状 [num_partitions, 3].
    """
    N, num_partitions = partitions_np.shape
    coefficients = []
    for i in range(num_partitions):
        weights = partitions_np[:, i]  # shape=[N,]
        W = onp.diag(weights)
        # 构造二阶项 X=[1, x, x^2], shape=(N,3)
        X = onp.vstack([onp.ones_like(x_np), x_np, x_np**2]).T  # (N,3)
        
        # 最小二乘: c = arg min ||W*(Xc - y)||^2
        c = onp.linalg.lstsq(W @ X, W @ y_np, rcond=None)[0]  # shape=[3,]
        coefficients.append(c)
    return onp.array(coefficients)  # shape=[num_partitions, 3]

##############################
# 3) RBF网络初始化
##############################
def init_rbf_params(rng_key, num_centers):
    """
    初始化 RBF 参数:
    centers, widths
    """
    key_centers, key_widths = jax.random.split(rng_key)
    centers = jax.random.normal(key_centers, shape=(num_centers,))
    widths = jnp.ones((num_centers,)) * 0.2
    return {"centers": centers, "widths": widths}

def rbf_forward(params, x):
    """
    RBF 前向, 归一化为 partition of unity
    partitions shape=[N, num_centers]
    """
    centers = params["centers"]
    widths = params["widths"]
    dist_sq = (x[:, None] - centers[None, :])**2
    rbf_vals = jnp.exp(- dist_sq / (widths**2))
    sum_rbf = jnp.sum(rbf_vals, axis=1, keepdims=True) + 1e-10
    partitions = rbf_vals / sum_rbf
    return partitions

##############################
# 4) 损失计算 (二阶多项式)
##############################
def compute_loss(params, x_jnp, y_jnp, num_partitions, lambda_reg):
    """
    1. 用 params => partitions
    2. 停梯度 -> onp 做二阶多项式拟合
    3. 合成 y_pou_approx
    4. loss = MSE + lambda_reg * ||partitions||^2
    """
    partitions = rbf_forward(params, x_jnp)  # [N,C]

    # 停梯度后转numpy, 做二阶回归
    partitions_np = onp.array(jax.lax.stop_gradient(partitions))
    x_np = onp.array(jax.lax.stop_gradient(x_jnp))
    y_np = onp.array(jax.lax.stop_gradient(y_jnp))

    coeffs = fit_local_polynomials_2nd_order(x_np, y_np, partitions_np)
    # coeffs shape=[C,3], c0=coeffs[:,0], c1=coeffs[:,1], c2=coeffs[:,2]

    c0 = jnp.array(coeffs[:, 0])
    c1 = jnp.array(coeffs[:, 1])
    c2 = jnp.array(coeffs[:, 2])

    # 合成逼近: c0 + c1*x + c2*x^2
    y_pou_approx = jnp.sum(
        partitions * (c0[None,:] + c1[None,:]*x_jnp[:,None] + c2[None,:]* (x_jnp[:,None]**2)),
        axis=1
    )
    # MSE
    mse = jnp.mean((y_pou_approx - y_jnp)**2)

    # L2 正则 (仅对partitions)
    l2_reg = jnp.linalg.norm(partitions, ord=2)
    loss = mse + lambda_reg * l2_reg
    return loss

loss_and_grad = jax.value_and_grad(compute_loss)

##############################
# 5) 两阶段训练
##############################
def train_two_phase_lsgd(params, x_train, y_train,
                         num_partitions=5,
                         num_epochs_phase1=1000,
                         num_epochs_phase2=1000,
                         lambda_reg=0.1,
                         lr_phase1=1e-2,
                         lr_phase2=5e-3):
    """
    与之前一样, 只不过这里是二阶多项式
    """
    opt1 = optax.sgd(lr_phase1)
    opt2 = optax.sgd(lr_phase2)

    opt_state1 = opt1.init(params)
    opt_state2 = opt2.init(params)

    x_jnp = jnp.array(x_train)
    y_jnp = jnp.array(y_train)

    cur_params = params

    # ============= Phase 1 =============
    for epoch in range(num_epochs_phase1):
        loss_val, grads = loss_and_grad(cur_params, x_jnp, y_jnp,
                                        num_partitions, lambda_reg)
        updates, opt_state1 = opt1.update(grads, opt_state1, cur_params)
        cur_params = optax.apply_updates(cur_params, updates)

        if epoch % 200 == 0:
            print(f"[Phase 1] Epoch {epoch}, Loss={loss_val: .6f}")

    params_phase1 = cur_params

    # ============= Phase 2 =============
    for epoch in range(num_epochs_phase2):
        loss_val, grads = loss_and_grad(cur_params, x_jnp, y_jnp,
                                        num_partitions, 0.0)
        updates, opt_state2 = opt2.update(grads, opt_state2, cur_params)
        cur_params = optax.apply_updates(cur_params, updates)

        if epoch % 200 == 0:
            print(f"[Phase 2] Epoch {epoch}, Loss={loss_val: .6f}")

    params_phase2 = cur_params
    return params_phase1, params_phase2

##############################
# 6) 评估和可视化
##############################
def fit_and_approx(params, x_data, y_data):
    """
    给定 params, x_data, y_data,
    先分区, 再做二阶多项式拟合, 最后合成 y_approx.
    返回 y_approx (onp array).
    """
    x_jnp = jnp.array(x_data)
    y_jnp = jnp.array(y_data)
    partitions = rbf_forward(params, x_jnp)

    partitions_np = onp.array(jax.lax.stop_gradient(partitions))
    coeffs = fit_local_polynomials_2nd_order(
        onp.array(x_data), onp.array(y_data), partitions_np
    )
    c0, c1, c2 = coeffs[:,0], coeffs[:,1], coeffs[:,2]

    # 合成逼近: c0 + c1*x + c2*x^2
    y_pou_approx = jnp.sum(
        partitions * (c0[None,:] + c1[None,:]*x_jnp[:,None] + c2[None,:]*(x_jnp[:,None]**2)),
        axis=1
    )
    return onp.array(y_pou_approx)

def visualize_partitions(params, x_data, title="Partition of Unity"):
    x_jnp = jnp.array(x_data)
    partitions = rbf_forward(params, x_jnp)
    partitions_np = onp.array(partitions)

    plt.figure(figsize=(6,4))
    for i in range(partitions_np.shape[1]):
        plt.plot(x_data, partitions_np[:,i], label=f"Partition {i+1}")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("weight")
    plt.legend()
    plt.show()

##############################
# 7) 主程序
##############################
def main():
    # 1) 生成训练数据 (三角波)
    N = 500
    x_train = onp.linspace(0, 1, N)
    y_train = triangle_wave(x_train, p=2)

    # 2) 初始化 RBF
    rng_key = jax.random.PRNGKey(42)
    num_partitions = 5
    params_init = init_rbf_params(rng_key, num_partitions)

    # 3) 两阶段训练 (二阶多项式)
    params_phase1, params_phase2 = train_two_phase_lsgd(
        params_init, x_train, y_train,
        num_partitions=num_partitions,
        num_epochs_phase1=1000,
        num_epochs_phase2=1000,
        lambda_reg=0.1,
        lr_phase1=1e-2,
        lr_phase2=5e-3
    )

    # ============= Phase 1 可视化 =============
    y_pred_1 = fit_and_approx(params_phase1, x_train, y_train)
    plt.figure(figsize=(6,4))
    plt.plot(x_train, y_train, '--', label="Ground Truth")
    plt.plot(x_train, y_pred_1, label="POU (2nd order) - PHASE 1")
    plt.legend(); plt.grid(True)
    plt.title("Final Approx (PHASE 1)")
    plt.show()

    visualize_partitions(params_phase1, x_train, "Partitions (PHASE 1)")

    # ============= Phase 2 可视化 =============
    y_pred_2 = fit_and_approx(params_phase2, x_train, y_train)
    plt.figure(figsize=(6,4))
    plt.plot(x_train, y_train, '--', label="Ground Truth")
    plt.plot(x_train, y_pred_2, label="POU (2nd order) - PHASE 2")
    plt.legend(); plt.grid(True)
    plt.title("Final Approx (PHASE 2)")
    plt.show()

    visualize_partitions(params_phase2, x_train, "Partitions (PHASE 2)")

if __name__ == "__main__":
    main()
