import os
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import matplotlib.pyplot as plt

# ----------------------------------
# 1. 超参数
# ----------------------------------
N_SUBDOMAINS = 2             # 子域个数(可根据需求修改)
N_COLLOCATION_POINTS = 1000  # 每个子域的训练点数
LEARNING_RATE = 1e-3
N_OPTIMIZATION_EPOCHS = 20000

key = jr.PRNGKey(42)

# 定义区间 [0,8]，每个子域长度:
delta = 8.0 / N_SUBDOMAINS

# ----------------------------------
# 2. PDE定义 + 精确解
# ----------------------------------
def phi(x):
    return (jnp.pi / 4.0) * x**2

def u_exact(x):
    return jnp.sin(phi(x))

def f_pde(x):
    # 对应于: d^2u/dx^2 + f_pde(x) = 0，(这里可能要根据你自己的 PDE 修改)
    return (jnp.pi**2 / 4.0) * x**2 * jnp.sin(phi(x)) - (jnp.pi / 2.0) * jnp.cos(phi(x))

# ----------------------------------
# 3. 定义子网络 (FBPINN)
# ----------------------------------
class FBPINN(eqx.Module):
    subnets: tuple

    def __init__(self, n_subdomains: int, key):
        keys = jr.split(key, n_subdomains)
        self.subnets = tuple(
            eqx.nn.MLP(
                in_size=1, out_size=1, width_size=20, depth=3,
                activation=jax.nn.tanh, key=keys[i]
            )
            for i in range(n_subdomains)
        )

# ----------------------------------
# 4. Window Function + PINN解拼接
# ----------------------------------
def window_function(x, i):
    """定义用于拼接子域输出的窗口函数。"""
    A = 30.0
    centers = jnp.linspace(0, 8, N_SUBDOMAINS)
    widths = delta * 2
    # 指定子域 i 的 Gauss 窗口
    w_i = jnp.exp(-A * (x - centers[i])**2 / widths**2)
    # 对所有子域窗口作归一化
    total_w = sum(
        jnp.exp(-A * (x - centers[j])**2 / widths**2)
        for j in range(N_SUBDOMAINS)
    )
    return w_i / total_w

@eqx.filter_jit
def net_eval(net, x):
    """
    原始错误是使用 eqx.filter_eval_shape(...) 得到的仅是 shape placeholder，
    这里直接使用 net(x) 得到实际输出。
    """
    x = jnp.atleast_1d(x).reshape(-1, 1)  # 确保 x 是 [batch, 1] 形式
    return net(x)[0, 0]  # 返回标量

def pinn_solution(x, net):
    """
    每个子域自身的 PINN 形式，可以根据边界条件或 PDE 特点进行修正。
    这里以 x*(8 - x)*网络输出 为例，保证边界 x=0/8 处解为 0。
    """
    return (x * (8.0 - x)) * net_eval(net, x)

def u_hat(x, params):
    """
    最终拼接：对每个子网络输出加权求和。
    """
    return sum(
        window_function(x, i) * pinn_solution(x, net_i)
        for i, net_i in enumerate(params.subnets)
    )

# ----------------------------------
# 5. PDE 残差 + 损失函数
# ----------------------------------
def pde_residual(x, net):
    """
    PDE: d^2u/dx^2 + f_pde(x) = 0
    """
    # 先计算 du/dx，再二次求导 d2u/dx^2
    dudx = jax.grad(lambda xx: pinn_solution(xx, net))(x)
    d2udx2 = jax.grad(lambda xx: dudx)(x)
    # PDE 残差
    return d2udx2 + f_pde(x)

def loss_fn(params, sub_colloc_points):
    # 1) PDE 残差项
    loss_pde = sum(
        jnp.mean(jax.vmap(lambda xx: pde_residual(xx, net))(x_i) ** 2)
        for net, x_i in zip(params.subnets, sub_colloc_points)
    )
    # 2) 边界条件损失 (u(0)=0, u(8)=0)
    loss_bc = u_hat(0.0, params)**2 + u_hat(8.0, params)**2

    return loss_pde + loss_bc

# ----------------------------------
# 6. 训练初始化
# ----------------------------------
model = FBPINN(N_SUBDOMAINS, key)
optimizer = optax.adam(LEARNING_RATE)
# 这里 eqx.filter(model, eqx.is_array) 就只获取数值权重(不包含其他 Python 对象)做优化
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

# 为每个子域分别采 collocation points
sampling_key = jr.PRNGKey(999)
all_colloc_points = [
    jr.uniform(jr.fold_in(sampling_key, i), (N_COLLOCATION_POINTS,),
               minval=i * delta, maxval=(i + 1) * delta)
    for i in range(N_SUBDOMAINS)
]

@eqx.filter_jit
def train_step(params, opt_state, sub_colloc_points):
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(params, sub_colloc_points)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, loss_val

# ----------------------------------
# 7. 训练循环
# ----------------------------------
loss_history = []
for it in range(N_OPTIMIZATION_EPOCHS):
    model, opt_state, loss_val = train_step(model, opt_state, all_colloc_points)

    if it % 2000 == 0:
        print(f"Iter={it}, loss={loss_val}")
    loss_history.append(loss_val)

# ----------------------------------
# 8. 可视化与误差计算
# ----------------------------------
os.makedirs("figures_subdomain", exist_ok=True)

x_plot = jnp.linspace(0.0, 8.0, 400)
u_pred = jax.vmap(lambda x: u_hat(x, model))(x_plot)
u_ref = u_exact(x_plot)

l1_error = jnp.mean(jnp.abs(u_pred - u_ref))
print(f"L1 Error: {l1_error:.6f}")

# 解曲线对比
plt.figure(figsize=(8, 5))
plt.plot(x_plot, u_ref, "--", label="Exact Solution")
plt.plot(x_plot, u_pred, label="FBPINN Solution")
plt.legend()
plt.grid(True)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title(f"FBPINN 1D PDE [0,8] - L1 Error: {l1_error:.6f}")
plt.savefig("figures_subdomain/solution.png", dpi=300)
plt.close()

# 损失曲线
plt.figure()
plt.plot(loss_history, label="Training Loss")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.savefig("figures_subdomain/training_loss.png", dpi=300)
plt.close()

print("Done. Plots saved in figures_subdomain")
