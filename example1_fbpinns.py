import jax
import jax.numpy as jnp
import optax
import equinox as eqx

def make_subdomain_batches(model, collocation_points):
    """
    根据每个子网的窗口函数，给每个子网筛选一批 collocation points。
    """
    subdomain_batches = []

    for i in range(model.num_subdomains):
        w = model.subdomain_window(i, collocation_points)  # 获取子域的窗口权重
        mask = w > 1e-5  # 只取窗口权重大于阈值的点
        pts = collocation_points[mask]  # 筛选出权重大于阈值的点
        subdomain_batches.append(pts)

    return subdomain_batches


@eqx.filter_value_and_grad
def loss_fn(model, subdomain_batches, omega):
    """
    Loss是所有子网loss加起来（pde residual loss）。
    每个子网只负责自己局部的点。
    """
    total_loss = 0.0

    for i in range(model.num_subdomains):
        x_i = subdomain_batches[i]
        if len(x_i) == 0:
            continue

        # 定义子域 i 的 u(x)
        def u_func(x):
            return model.subdomain_pred(i, x).squeeze()  # shape (,) for scalar output

        # 求导
        dudx = jax.vmap(jax.grad(u_func))(x_i)

        # 计算残差（比如这里是u'(x) = cos(omega x)）
        residual = dudx - jnp.cos(omega * x_i)

        total_loss += jnp.mean(residual**2)

    return total_loss


@eqx.filter_jit
def train_step(model, opt_state, subdomain_batches, optimizer, omega):
    """
    单步训练：计算pde残差loss，更新参数。
    """
    loss, grads = loss_fn(model, subdomain_batches, omega)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def train_fbpinn(model, collocation_points, omega, batch_size=32, steps=10000, lr=1e-3, x_test=None, u_exact_fn=None):
    """
    真正的训练主程序，loss基于pde残差。
    现在支持：每次print时也计算test的L1误差。
    """
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    loss_list = []
    l1_list = []
    l1_steps = []

    for step in range(steps):
        # 每次从collocation_points中随机选取一个batch
        key = jax.random.PRNGKey(step)
        shuffle_idx = jax.random.permutation(key, len(collocation_points))
        
        # 获取一个小批量（batch_size）
        batch_idx = shuffle_idx[:batch_size]
        batch_points = collocation_points[batch_idx]

        # 重建子网的batch
        subdomain_batches = make_subdomain_batches(model, batch_points)

        # 训练一步
        model, opt_state, loss = train_step(model, opt_state, subdomain_batches, optimizer, omega)
        loss_list.append(float(loss))

        # 每500步或最后一步，打印+计算test L1
        if step % 100 == 0 or step == steps - 1:
            if x_test is not None and u_exact_fn is not None:
                u_pred = inference(model, x_test)
                u_true = u_exact_fn(x_test)
                l1_error = jnp.mean(jnp.abs(u_pred - u_true))
                l1_list.append(float(l1_error))
                l1_steps.append(step)
                print(f"Step {step}, Loss={loss:.4e}, Test L1 Error={l1_error:.4e}")
            else:
                print(f"Step {step}, Loss={loss:.4e}")

    return model, jnp.array(loss_list), (jnp.array(l1_steps), jnp.array(l1_list))



def inference(model, x):
    """
    推理阶段，所有子网输出乘窗口再加权归一化。
    相当于： u(x) = sum_i w_i(x) * u_i(x) / sum_i w_i(x)
    """
    u_total = 0.0
    w_total = 0.0

    for i in range(model.num_subdomains):
        out = model.subdomain_pred(i, x)  # 每个子网的预测输出
        w = model.subdomain_window(i, x)  # 每个子网的窗口权重
        u_total += out * w
        w_total += w

    return u_total / (w_total + 1e-8)  # 防止除以0


if __name__ == "__main__":
    from model.fbpinn_model import FBPINN
    from physics.pde_cosine import ansatz, u_exact
    import math
    # Initialization
    mlp_config = {
        "in_size": 1,
        "out_size": 1,
        "width_size": 16,
        "depth": 2,
        "activation": jax.nn.tanh
    }
    
    domain = (-2 * math.pi, 2 * math.pi)

    n_sub = 5
    overlap = 4
    steps = 3000
    lr = 1e-3
    n_points_per_subdomain = 20

    # We can define the subdomain list manually here!
    total_len = domain[1] - domain[0]
    step_size = total_len / n_sub
    width = step_size + overlap

    centers = jnp.linspace(domain[0] + step_size / 2,
                           domain[1] - step_size / 2,
                           n_sub)
    # subdomains_list = [ (left_i, right_i), ... ] a tuple list!
    subdomains_list = []
    for i in range(n_sub):
        left = float(centers[i] - width / 2)
        right = float(centers[i] + width / 2)
        subdomains_list.append((left, right))
        
    model = FBPINN(
        key=jax.random.PRNGKey(0),
        num_subdomains=5,
        ansatz=ansatz,
        subdomains=subdomains_list,
        mlp_config=mlp_config
    )

    # collocation points
    domain = (-2 * jnp.pi, 2 * jnp.pi)
    key = jax.random.PRNGKey(42)
    collocation_points = jax.random.uniform(key, (300,), minval=domain[0], maxval=domain[1])

    # PDE参数，比如 cos(omega x)
    omega = 1.0
    x_test = jnp.linspace(domain[0], domain[1], 300)
    # 开始训练
    model, losses, (test_steps, test_l1_errors) = train_fbpinn(
        model,
        collocation_points,
        omega=omega,
        batch_size=32,
        steps=30000,
        lr=1e-3,
        x_test=x_test,
        u_exact_fn=u_exact
    )

    # 推理
    u_pred = inference(model, x_test)

    # 计算真解
    u_true = u_exact(x_test)

    # 比较误差，比如L2误差或者L1误差
    l2_error = jnp.mean((u_pred - u_true)**2)
    l1_error = jnp.mean(jnp.abs(u_pred - u_true))

    print(f"L2 error = {l2_error:.4e}")
    print(f"L1 error = {l1_error:.4e}")
    import matplotlib.pyplot as plt

    # 画图：模型预测 vs 真解
    plt.figure(figsize=(8,5))
    plt.plot(x_test, u_true, label="Exact $u(x)$", linestyle='--')
    plt.plot(x_test, u_pred, label="Predicted $\hat{u}(x)$", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.title(f"FBPINN Prediction vs Exact\nL2 error={l2_error:.2e}, L1 error={l1_error:.2e}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
