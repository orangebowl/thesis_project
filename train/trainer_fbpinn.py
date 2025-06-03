import os
import sys
import yaml
import importlib
import jax
import jax.numpy as jnp
import optax
import equinox as eqx

# add path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import the model
from model.fbpinn_model import FBPINN
# import some data utils functions
from utils.data_utils import generate_collocation, generate_subdomains
# import some visualization functions
from utils.visualizer import (
    plot_prediction_vs_exact,
    plot_training_loss,
    plot_test_l1_curve,
    plot_window_weights,
    save_training_stats,
    plot_subdomain_partials
)
from tqdm import trange


# train_fbpinn_global.py
import os, jax, jax.numpy as jnp, optax, equinox as eqx
from tqdm import trange


@eqx.filter_jit
def _step(model, opt_state, colloc_full, optimizer):
    if colloc_full.shape[0] == 0:
        return model, opt_state, jnp.array(0.0)

    # 直接用 residual_fn 计算 loss
    def loss_fn(m):
        res_sq = jax.vmap(m.residual_fn, (None, 0))(m, colloc_full)
        return jnp.mean(res_sq)

    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val


# ------------------ L1 误差评估 ------------------ #
@eqx.filter_jit
def compute_l1(model, x_test, u_test_exact):
    pred = model(x_test).squeeze()
    return jnp.mean(jnp.abs(pred - u_test_exact.squeeze()))


# ------------------- 主训练循环 ------------------- #
def train_fbpinn(
    *,
    model:       eqx.Module,
    colloc_full: jax.Array,            # (N_total, xdim) —— 全域采样
    steps:       int,
    lr:          float,
    x_test:      jax.Array  = None,
    u_exact:     callable   = None,
    save_dir:    str        = None,
    checkpoint_every: int   = 0,
):
    """FBPINN 训练（全域 collocation 版）"""
    # 1) 优化器
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # 2) 历史记录
    loss_hist, l1_hist, l1_steps = [], [], []
    pbar = trange(steps, desc="FBPINN", dynamic_ncols=True)

    # 3) 训练循环
    for step in pbar:
        model, opt_state, loss = _step(model, opt_state,
                                       colloc_full, optimizer)
        loss_val = float(loss); loss_hist.append(loss_val)

        # 定期评估 L1
        if x_test is not None and (step % 100 == 0 or step == steps - 1):
            u_test_exact = u_exact(x_test)
            l1 = float(compute_l1(model, x_test, u_test_exact))
            l1_hist.append(l1); l1_steps.append(step)
            pbar.set_postfix(loss=f"{loss_val:.2e}", l1=f"{l1:.2e}")
        else:
            pbar.set_postfix(loss=f"{loss_val:.2e}")

        # checkpoint
        if checkpoint_every and save_dir and (step + 1) % checkpoint_every == 0:
            os.makedirs(save_dir, exist_ok=True)
            eqx.tree_serialise_leaves(
                os.path.join(save_dir, f"ckpt_{step + 1}.eqx"), model
            )

    return model, jnp.array(loss_hist), (jnp.array(l1_steps), jnp.array(l1_hist))
