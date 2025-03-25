import jax
import jax.numpy as jnp
import optax
import equinox as eqx


def loss_fbpinn(model, x_collocation, pde_residual, window_fn):
    """
    通用 FBPINN Loss 计算：基于 residual 和窗口函数。
    """
    residuals = jax.vmap(lambda x: pde_residual(model, x, window_fn))(x_collocation)
    return jnp.mean(residuals ** 2)


@eqx.filter_jit
def train_step(model, opt_state, x_collocation, optimizer, pde_residual, window_fn):
    """
    单步梯度更新：自动计算 loss 和梯度。
    """
    loss_val, grads = eqx.filter_value_and_grad(loss_fbpinn)(model, x_collocation, pde_residual, window_fn)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val


def train_fbpinn(model, x_collocation, steps, lr, pde_residual, window_fn):
    """
    通用 FBPINN 训练主函数
    参数：
      - model: FBPINN 模型（SmoothFBPINN）
      - x_collocation: 全局 collocation 点
      - steps: 训练步数
      - lr: 学习率
      - pde_residual: 外部传入的 residual 函数
      - window_fn: 外部传入的窗口函数（Partition of Unity）
    返回：
      - model: 训练后的模型
      - loss_list: 每步 loss 记录
    """
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    loss_list = []

    for i in range(steps):
        model, opt_state, loss_val = train_step(model, opt_state, x_collocation, optimizer, pde_residual, window_fn)
        loss_list.append(loss_val)

        if i % 1000 == 0:
            print(f"[FBPINN] Step={i}, Loss={loss_val:.3e}")

    return model, loss_list
