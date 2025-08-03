import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from tqdm import trange
from typing import Callable, Optional

def train_fbpinn(
    *, 
    key: jax.random.PRNGKey, 
    model: eqx.Module, 
    problem,
    colloc: jax.Array, 
    lr: float = 3e-4, 
    steps: int = 10_000,
    x_test: Optional[jax.Array] = None,
    u_exact: Optional[Callable[[jax.Array], jax.Array]] = None,
    eval_every: int = 100
):

    colloc = colloc.astype(jnp.float32)

    # 将模型分区为可训练参数 (params) 和静态数据 (static)
    params, static = eqx.partition(model, eqx.is_array)

    # 辅助函数，用于根据当前参数重建模型
    def build_model(p):
        return eqx.combine(p, static)

    # 损失函数现在只需要可训练参数，因为`static`已从外部作用域捕获
    def loss_fn(p, xy):
        return problem.residual(build_model(p), xy)

    @eqx.filter_jit
    def eval_fn(p):
        """JIT编译的评估函数，用于在测试集上计算L1误差。"""
        if x_test is None or u_exact is None:
            return jnp.nan
        
        full_model = build_model(p)
        pred = full_model(x_test.astype(jnp.float32)).squeeze()
        exact = u_exact(x_test).squeeze()
        return jnp.mean(jnp.abs(pred - exact))

    # 初始化 Adam 优化器
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @eqx.filter_jit
    def step_fn(p, o, xy_full):
        """
        在整个数据集上执行单次JIT编译的训练步骤。
        """
        # 如果数据是按子域划分的，则将其展平以便处理
        if xy_full.ndim > 2:
            xy_full = xy_full.reshape(-1, xy_full.shape[-1])
        
        # 一次性在整个数据集上计算损失和梯度
        loss_val, grads = jax.value_and_grad(loss_fn)(p, xy_full)
        
        # 应用优化器更新
        updates, o = optimizer.update(grads, o, p)
        p = eqx.apply_updates(p, updates)
        
        return p, o, loss_val

    # --- 训练循环 ---
    print("JIT 编译完整模型的训练步骤... (这可能需要很长时间)")
    
    # 使用一小部分数据预编译，以避免在编译期间进行大的内存分配
    # 结构是相同的，所以编译好的函数将适用于完整数据
    _colloc_flat = colloc.reshape(-1, colloc.shape[-1])
    if _colloc_flat.shape[0] > 0:
        step_fn(params, opt_state, _colloc_flat[:1])
    print("编译完成。")

    loss_history, l1_history, l1_steps = [], [], []
    
    bar = trange(steps, dynamic_ncols=True, desc="FBPINN 训练 (All Active)")
    for s in bar:
        params, opt_state, loss_val = step_fn(params, opt_state, colloc)
        loss_history.append(float(loss_val))

        # 周期性地评估并记录L1误差
        if (s + 1) % eval_every == 0 or s + 1 == steps:
            l1_error = float(eval_fn(params))
            l1_history.append(l1_error)
            l1_steps.append(s + 1)
            bar.set_postfix(loss=f"{loss_val:.3e}", L1=f"{l1_error:.3e}")
        else:
            bar.set_postfix(loss=f"{loss_val:.3e}")

    # 返回最终训练好的模型和记录的历史数据
    final_model = build_model(params)
    return final_model, jnp.array(loss_history), (jnp.array(l1_steps), jnp.array(l1_history))
