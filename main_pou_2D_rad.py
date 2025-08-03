# %matplotlib widget
from __future__ import annotations
import jax, jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from functools import partial
import os, sys
import equinox as eqx
import optax
from typing import Sequence, Dict, Tuple, Any, Callable, Optional
from tqdm import trange
import numpy as np
import dataclasses

# --- 项目根目录设置 ---
project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.append(project_root)

# ==============================================================================
# 1. 从您的本地文件导入核心模块
# ==============================================================================
# 假设您有一个名为 physics 的文件夹，其中包含 problems.py
from physics.problems import FirstOrderFreq68,Wave1DHighFreq,Poisson2D_freq66,FirstOrderFreq1010,ViscousBurgersFBPINN
# 假设您有一个名为 model 的文件夹，其中包含 fbpinn_model.py
from model.fbpinn_model import FBPINN_PoU
# 假设您有一个名为 utils 的文件夹，其中包含 data_utils.py
from utils.data_utils import generate_collocation


# --- 为了让代码可独立运行，这里提供上述模块的模拟实现 ---


# ==============================================================================
# 2. PoU Learning Modules
# ==============================================================================
class RBFPOUNet:
    def __init__(self, input_dim: int, num_centers: int, domain, key=None):
        self.input_dim, self.num_experts = input_dim, num_centers
        key = jax.random.PRNGKey(42) if key is None else key
        min_b, max_b = jnp.array(domain[0]), jnp.array(domain[1])
        base_centers = jnp.linspace(min_b, max_b, num_centers)
        jitters = 0.05 * (max_b - min_b) * jax.random.normal(key, base_centers.shape)
        self._init_centers = base_centers + jitters
        self._init_widths  = 0.5 * (max_b - min_b) / num_centers * jnp.ones((num_centers,))
    def init_params(self):
        return {"centers": self._init_centers, "widths":  self._init_widths}
    @staticmethod
    def forward(params, x):
        c, w = params["centers"], params["widths"]
        d2 = jnp.sum((x[:, None, :] - c[None, :, :])**2, -1); log_phi = -d2 / (w**2 + 1e-12)
        log_phi = log_phi - jnp.max(log_phi, 1, keepdims=True); phi = jnp.exp(log_phi)
        phi = jnp.maximum(phi, 1e-8); return phi / jnp.sum(phi, 1, keepdims=True)

def glorot(key, shape):
    return jax.random.uniform(key, shape, minval=-jnp.sqrt(6/(shape[0]+shape[1])), maxval=jnp.sqrt(6/(shape[0]+shape[1])))

def init_mlp_1d(key, hidden: Sequence[int], out_dim: int) -> Dict[str, Any]:
    params = {}; dims = [1] + list(hidden) + [out_dim]; keys = jax.random.split(key, len(dims) - 1)
    for i, (m, n) in enumerate(zip(dims[:-1], dims[1:])):
        params[f'W{i}'] = glorot(keys[i], (m, n)); params[f'b{i}'] = jnp.zeros((n,))
    return params

def mlp_forward_1d(params: Dict[str, Any], x: jnp.ndarray) -> jnp.ndarray:
    h = x; n_layer = (len(params) // 2) - 1
    for i in range(n_layer): h = jnp.tanh(h @ params[f'W{i}'] + params[f'b{i}'])
    return h @ params[f'W{n_layer}'] + params[f'b{n_layer}']

class SepMLPPOUNet:
    def __init__(self, nx: int, ny: int, hidden: Sequence[int] = (32, 32), tau: float = 0.1, key = jax.random.PRNGKey(0)):
        self.nx, self.ny = nx, ny; self.num_experts = nx * ny; self.tau = tau
        kx, ky = jax.random.split(key)
        self.param_x = init_mlp_1d(kx, hidden, nx); self.param_y = init_mlp_1d(ky, hidden, ny)
    def init_params(self) -> Dict[str, Any]:
        return {'x': self.param_x, 'y': self.param_y}
    def forward(self, params: Dict[str, Any], xy: jnp.ndarray) -> jnp.ndarray:
        xy = jnp.atleast_2d(xy); x, y = xy[:, :1], xy[:, 1:]
        z_x = mlp_forward_1d(params['x'], x); z_y = mlp_forward_1d(params['y'], y)
        logits = (z_x[:, :, None] + z_y[:, None, :]) / self.tau
        logits_flat = logits.reshape(x.shape[0], -1)
        logits_stable = logits_flat - jnp.max(logits_flat, axis=-1, keepdims=True)
        return jax.nn.softmax(logits_stable, axis=-1)

class WindowModule(eqx.Module):
    pou_net: Any = eqx.static_field(); params: Dict
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray: return self.pou_net.forward(self.params, x)

# ==============================================================================
# 3. Local Fitting and PoU Training
# ==============================================================================
def _design_matrix(x: jnp.ndarray) -> jnp.ndarray:
    d = x.shape[-1]
    if d == 1: x_flat = x.squeeze(-1) if x.ndim > 1 else x; return jnp.stack([jnp.ones_like(x_flat), x_flat, x_flat**2], -1)
    elif d == 2: x1, x2 = x[:, 0], x[:, 1]; return jnp.stack([jnp.ones_like(x1), x1, x2, x1**2, x1*x2, x2**2], -1)
    else: raise ValueError("Only 1-D or 2-D supported")

def fit_local_polynomials(x, y, w, lam: float = 0.0):
    A, y = _design_matrix(x), y[:, None]; k = A.shape[-1]
    def _solve(weights):
        Aw = A * weights[:, None]; M  = A.T @ Aw; b  = (Aw.T @ y).squeeze(-1)
        return jnp.linalg.solve(M + lam*jnp.eye(k), b)
    return jax.vmap(_solve, 1, 0)(w)

def _predict_from_coeffs(x, coeffs, partitions):
    A = _design_matrix(x); y_cent = A @ coeffs.T; return jnp.sum(partitions * y_cent, 1)

@dataclasses.dataclass
class LSGDConfig:
    n_epochs: int = 15000; lr: float = 1e-4; lam_init: float = 0.01; rho: float = 0.99; n_stag: int = 300

def run_lsgd(pou_net, initial_params, x, y, cfg: LSGDConfig):
    params = initial_params
    def loss_fn(p, lam):
        part = pou_net.forward(p, x); coeffs = fit_local_polynomials(x, y, part, lam)
        pred = _predict_from_coeffs(x, coeffs, part); return jnp.mean((pred - y)**2)
    valgrad_fn = jax.jit(jax.value_and_grad(loss_fn))
    opt = optax.adam(cfg.lr); opt_state = opt.init(params); lam = jnp.array(cfg.lam_init); best, stag = jnp.inf, 0
    bar = trange(cfg.n_epochs, desc=f"PoU-LSGD (N={pou_net.num_experts})", dynamic_ncols=True)
    for ep in bar:
        loss_val, grads = valgrad_fn(params, lam); updates, opt_state = opt.update(grads, opt_state)
        params = eqx.apply_updates(params, updates); bar.set_postfix(loss=f"{loss_val:.3e}")
        if loss_val < best - 1e-12: best, stag = loss_val, 0
        else: stag += 1
        if stag > cfg.n_stag: lam *= cfg.rho; stag = 0; print(f"\nStagnation. Lambda decayed to {lam:.4f}")
    return params

# ==============================================================================
# 4. RAD Sampling Core Functions
# ==============================================================================
@partial(jax.jit, static_argnames=['problem', 'static'])
def pointwise_residual(problem, params, static, x):
    model = eqx.combine(params, static)
    raw_res = problem.pointwise_residual(model, x)
    return jnp.abs(raw_res.squeeze())

def rad_sample(key, problem, params, static, *, domain, n_draw, pool_size, k=3.0, c=1.0):
    lo, hi = jnp.array(domain[0]), jnp.array(domain[1])
    problem_dim = len(lo)
    pool = jax.random.uniform(key, (pool_size, problem_dim), minval=lo, maxval=hi)
    key, sub_key = jax.random.split(key)
    res_vals = pointwise_residual(problem, params, static, pool)
    prob = res_vals**k / jnp.mean(res_vals**k) + c
    prob = prob / prob.sum()
    idx = jax.random.choice(sub_key, pool_size, (n_draw,), p=prob, replace=False)
    return pool[idx]

# ==============================================================================
# 5. Two Independent Training Functions (Modified for RAD)
# ==============================================================================
def train_pinn_with_batching(
    key: jax.Array,
    model: eqx.Module,
    problem: BaseProblem,
    colloc: jax.Array,
    lr: float,
    steps: int,
    batch_size: int,
    x_test: jax.Array,
    u_exact: jax.Array,
    *,
    eval_every: int = 100,
    rad_config: Optional[Dict[str, Any]] = None,
):
    """Returns (final_model, best_l1, history_tuple, current_colloc)."""
    params, static = eqx.partition(model, eqx.is_array)
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @eqx.filter_jit
    def loss_fn(p, xy):
        return problem.residual(eqx.combine(p, static), xy)

    @eqx.filter_jit
    def step_fn(p, o, batch):
        loss, g = jax.value_and_grad(loss_fn)(p, batch)
        up, o = opt.update(g, o)
        p = eqx.apply_updates(p, up)
        return p, o, loss

    @eqx.filter_jit
    def eval_fn(p):
        m = eqx.combine(p, static)
        pred = jax.vmap(m)(x_test).squeeze()
        return jnp.mean(jnp.abs(pred - u_exact.squeeze()))

    # JIT warm-up
    step_fn(params, opt_state, colloc[:batch_size])

    loss_hist, l1_hist, l1_steps = [], [], []
    best_l1, best_params = np.inf, params
    current_colloc = colloc

    bar = trange(steps, desc="PINN (N=1)", dynamic_ncols=True)
    for s in bar:
        # ---------- RAD 全量重采样 ----------
        if rad_config and s and (s % rad_config["resample_every"] == 0):
            key, rad_key = jax.random.split(key)
            sp = rad_config["sample_params"].copy()
            sp["n_draw"] = len(current_colloc)
            cur_par, _ = eqx.partition(eqx.combine(params, static), eqx.is_array)
            current_colloc = rad_sample(rad_key, problem, cur_par, static, **sp)

        key, sub = jax.random.split(key)
        idx = jax.random.choice(sub, len(current_colloc), (batch_size,), replace=False)
        params, opt_state, loss_val = step_fn(params, opt_state, current_colloc[idx])
        loss_hist.append(float(loss_val))

        if (s + 1) % eval_every == 0 or (s + 1) == steps:
            l1 = float(eval_fn(params))
            l1_hist.append(l1)
            l1_steps.append(s + 1)
            if l1 < best_l1:
                best_l1, best_params = l1, params
            bar.set_postfix(loss=f"{loss_val:.3e}", L1_err=f"{l1:.3e}")
        else:
            bar.set_postfix(loss=f"{loss_val:.3e}")

    history = (jnp.array(loss_hist), jnp.array(l1_steps), jnp.array(l1_hist))
    final_model = eqx.combine(best_params, static)
    return final_model, best_l1, history, current_colloc

# ---------------------------------------------------------------------------
# 2.  full-batch FBPINN 训练（n_sub > 1）
# ---------------------------------------------------------------------------
def train_fbpinn_no_batching(
    key: jax.Array,
    model: eqx.Module,
    problem: BaseProblem,
    colloc: jax.Array,
    lr: float,
    steps: int,
    x_test: jax.Array,
    u_exact: jax.Array,
    *,
    eval_every: int = 100,
    rad_config: Optional[Dict[str, Any]] = None,
):
    """Returns (final_model, best_l1, history_tuple, current_colloc)."""
    params, static = eqx.partition(model, eqx.is_array)
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @eqx.filter_jit
    def loss_fn(p, xy):
        return problem.residual(eqx.combine(p, static), xy)

    @eqx.filter_jit
    def step_fn(p, o, full_batch):
        loss, g = jax.value_and_grad(loss_fn)(p, full_batch)
        up, o = opt.update(g, o)
        p = eqx.apply_updates(p, up)
        return p, o, loss

    @eqx.filter_jit
    def eval_fn(p):
        m = eqx.combine(p, static)
        pred = jax.vmap(m)(x_test).squeeze()
        return jnp.mean(jnp.abs(pred - u_exact.squeeze()))

    # JIT warm-up
    step_fn(params, opt_state, colloc)

    loss_hist, l1_hist, l1_steps = [], [], []
    best_l1, best_params = np.inf, params
    current_colloc = colloc
    n_sub = len(model.subnets)

    bar = trange(steps, desc=f"FBPINN (N={n_sub})", dynamic_ncols=True)
    for s in bar:
        if rad_config and s and (s % rad_config["resample_every"] == 0):
            key, rad_key = jax.random.split(key)
            sp = rad_config["sample_params"].copy()
            sp["n_draw"] = len(current_colloc)
            cur_par, _ = eqx.partition(eqx.combine(params, static), eqx.is_array)
            current_colloc = rad_sample(rad_key, problem, cur_par, static, **sp)

        params, opt_state, loss_val = step_fn(params, opt_state, current_colloc)
        loss_hist.append(float(loss_val))

        if (s + 1) % eval_every == 0 or (s + 1) == steps:
            l1 = float(eval_fn(params))
            l1_hist.append(l1)
            l1_steps.append(s + 1)
            if l1 < best_l1:
                best_l1, best_params = l1, params
            bar.set_postfix(loss=f"{loss_val:.3e}", L1_err=f"{l1:.3e}")
        else:
            bar.set_postfix(loss=f"{loss_val:.3e}")

    history = (jnp.array(loss_hist), jnp.array(l1_steps), jnp.array(l1_hist))
    final_model = eqx.combine(best_params, static)
    return final_model, best_l1, history, current_colloc

# ==============================================================================
# 6. Analysis and Plotting Functions
# ==============================================================================
def check_territory(window_fn: Callable, x_test: jax.Array, dominance_threshold: float = 0.8) -> bool:
    weights = window_fn(x_test); num_partitions = weights.shape[1]
    if num_partitions == 1: return True
    dominant_partition_indices = jnp.argmax(weights, axis=1); unique_dominant_indices = jnp.unique(dominant_partition_indices)
    if len(unique_dominant_indices) < num_partitions:
        missing_indices = jnp.setdiff1d(jnp.arange(num_partitions), unique_dominant_indices)
        print(f"       - HEALTH CHECK FAILED: Partitions {missing_indices} have no territory!"); return False
    for i in range(num_partitions):
        weights_in_territory = weights[dominant_partition_indices == i, i]
        if weights_in_territory.size == 0: continue
        if jnp.max(weights_in_territory) < dominance_threshold:
            print(f"       - HEALTH CHECK FAILED: Partition {i} is weak (peak weight {jnp.max(weights_in_territory):.3e} < {dominance_threshold})."); return False
    print(f"       - Health check passed for {num_partitions} partitions."); return True

def get_grid_dims(n_sub: int) -> Tuple[int, int]:
    if n_sub == 1: return 1, 1
    best_factor = 1
    for i in range(2, int(jnp.sqrt(n_sub)) + 1):
        if n_sub % i == 0: best_factor = i
    return (best_factor, n_sub // best_factor)

def plot_results(model, problem, x_test, u_exact, stage_index, n_sub, save_dir, test_n_2d=100):
    problem_dim = len(problem.domain[0])
    print(f"   -> Stage {stage_index}: Generating FBPINN solution plots (n_sub={n_sub})...")
    u_pred = jax.vmap(model)(x_test); u_pred = u_pred.reshape(-1).T
    u_ex = u_exact.reshape(-1)
    if u_pred.shape != u_ex.shape:
        u_pred, u_ex = u_pred.squeeze(), u_ex.squeeze()
    error = jnp.abs(u_pred - u_ex)
    if problem_dim == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6)); ax.plot(x_test, u_ex, 'k-', label='Exact', linewidth=2); ax.plot(x_test, u_pred, 'r--', label='Predicted')
        ax.set_title(f'Stage {stage_index}: FBPINN Result ({n_sub} subdomains)'); ax.set_xlabel('x'); ax.set_ylabel('u(x)'); ax.legend(); ax.grid(True)
    else: # 2D
        u_pred_grid = u_pred.reshape(test_n_2d, test_n_2d); u_ex_grid = u_ex.reshape(test_n_2d, test_n_2d); error_grid = error.reshape(test_n_2d, test_n_2d)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5)); domain_extent = [problem.domain[0][0], problem.domain[1][0], problem.domain[0][1], problem.domain[1][1]]
        vmax, vmin = jnp.max(u_ex_grid), jnp.min(u_ex_grid)
        im1 = axes[0].imshow(u_pred_grid.T, extent=domain_extent, origin='lower', cmap='viridis', vmax=vmax, vmin=vmin); axes[0].set_title('Predicted'); fig.colorbar(im1, ax=axes[0])
        im2 = axes[1].imshow(u_ex_grid.T, extent=domain_extent, origin='lower', cmap='viridis', vmax=vmax, vmin=vmin); axes[1].set_title('Exact'); fig.colorbar(im2, ax=axes[1])
        im3 = axes[2].imshow(error_grid.T, extent=domain_extent, origin='lower', cmap='Reds'); axes[2].set_title('Absolute L1 Error'); fig.colorbar(im3, ax=axes[2])
        fig.suptitle(f'Stage {stage_index}: FBPINN Result ({n_sub} subdomains)', fontsize=16); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filepath = os.path.join(save_dir, f"fbpinn_stage_{stage_index}_nsub_{n_sub}.png"); plt.savefig(filepath, dpi=300); plt.close(fig)
    print(f"           ... plots saved to {filepath}")

def plot_pou_results(window_fn, problem, x_test, stage_index, n_sub_next, save_dir, test_n_2d=100):
    problem_dim = len(problem.domain[0]); print(f"   -> Stage {stage_index}: Generating PoU partition plots for next stage (n_sub={n_sub_next})...")
    weights = window_fn(x_test); num_windows = weights.shape[1]
    if problem_dim == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for i in range(weights.shape[1]): ax.plot(x_test, weights[:, i], label=f'Window {i+1}')
        ax.set_title(f'Stage {stage_index}: Learned PoU for {n_sub_next} Subdomains'); ax.set_xlabel('x'); ax.set_ylabel('Weight'); ax.legend(); ax.grid(True); ax.set_ylim(-0.1, 1.1)
    else:
        nx, ny = get_grid_dims(num_windows)
        fig, axes = plt.subplots(ny, nx, figsize=(4 * nx, 3.5 * ny), squeeze=False); axes = axes.ravel(); domain_extent = [problem.domain[0][0], problem.domain[1][0], problem.domain[0][1], problem.domain[1][1]]
        for i in range(num_windows):
            ax = axes[i]; window_grid = weights[:, i].reshape(test_n_2d, test_n_2d); im = ax.imshow(window_grid.T, extent=domain_extent, origin='lower', cmap='inferno', vmin=0, vmax=1)
            ax.set_title(f'Window {i+1}'); fig.colorbar(im, ax=ax)
        for j in range(num_windows, len(axes)): axes[j].axis('off')
        fig.suptitle(f'Stage {stage_index}: Learned PoU for {n_sub_next} Subdomains', fontsize=16); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filepath = os.path.join(save_dir, f"pou_stage_{stage_index}_nsub_next_{n_sub_next}.png"); plt.savefig(filepath, dpi=300); plt.close(fig)
    print(f"           ... plots saved to {filepath}")

def plot_loss_history(loss_hist, l1_steps, l1_hist, stage_index, n_sub, save_dir):
    print(f"   -> Stage {stage_index}: Generating loss history plot (n_sub={n_sub})...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6)); ax.plot(loss_hist, label='Training Loss (MSE)', color='tab:blue', alpha=0.8); ax.set_yscale('log'); ax.set_xlabel('Steps'); ax.set_ylabel('Loss', color='tab:blue'); ax.grid(True, which="both", ls="--", alpha=0.5)
    ax2 = ax.twinx(); ax2.plot(l1_steps, l1_hist, 'o-', label='Evaluation Metric (L1 Error)', color='tab:red'); ax2.set_ylabel('Relative L1 Error', color='tab:red')
    fig.suptitle(f'Stage {stage_index}: History (n_sub={n_sub})', fontsize=16); lines, labels = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best'); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filepath = os.path.join(save_dir, f"loss_history_stage_{stage_index}_nsub_{n_sub}.png"); plt.savefig(filepath, dpi=300); plt.close(fig)
    print(f"           ... plot saved to {filepath}")

def save_npz_history(save_dir, stage_index, n_sub, history_data):
    loss_hist, l1_steps, l1_hist = history_data
    filename = f"history_stage_{stage_index}_nsub_{n_sub}.npz"
    filepath = os.path.join(save_dir, filename); print(f"   -> Saving training history to {filepath}...")
    np.savez_compressed(filepath, loss_hist=np.asarray(loss_hist), l1_steps=np.asarray(l1_steps), l1_hist=np.asarray(l1_hist))

def plot_colloc_points(colloc, domain, stage_id, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.scatter(colloc[:, 0], colloc[:, 1], s=5, alpha=0.6)
    plt.xlim(domain[0][0], domain[1][0])
    plt.ylim(domain[0][1], domain[1][1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Collocation Points (Stage: {stage_id}, N={colloc.shape[0]})")
    filepath = os.path.join(save_dir, f"rad_colloc_stage_{stage_id}.png")
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   -> RAD plot saved to {filepath}")


def run_hierarchical_fbpinn(key: jax.Array,
                            problem: BaseProblem,
                            config: Dict[str, Any]):
    """
    PINN ➜ PoU-LSGD ➜ FBPINN  (RAD)  整体流程
    训练完每一阶段后自动生成结果图 / 历史图 / PoU 图 / RAD 点图。
    返回  {n_sub : {'rel_l1_error': …}, …}
    """
    # --- 数据 ---------------------------------------------------------------
    domain = problem.domain
    test_n_2d  = config.get("test_n_2d", 101)
    colloc_n_2d = config.get("colloc_n_2d", 100)
    x_test      = jnp.asarray(generate_collocation(domain, test_n_2d,  strategy="grid"))
    colloc_init = jnp.asarray(generate_collocation(domain, colloc_n_2d, strategy="grid"))
    u_exact     = jax.vmap(problem.exact)(x_test).squeeze()
    save_dir    = config["save_dir"]

    # 结果字典
    results: Dict[int, Dict[str, float]] = {}

    # ======================  Stage-0 :  单域 PINN  ===========================
    key, subkey = jax.random.split(key)
    model = FBPINN_PoU(
        key=subkey,
        num_subdomains=1,
        mlp_config=config["mlp_conf"],
        domain=domain,
        ansatz=problem.ansatz,
        residual_fn=problem.residual,
    )

    model, best_l1, hist, colloc_cur = train_pinn_with_batching(
        subkey, model, problem, colloc_init,
        config["FBPINN_LR"], config["FBPINN_STEPS"], config["BATCH_SIZE"],
        x_test, u_exact, rad_config=config.get("rad_config"),
    )

    # —— 画图
    loss_hist, l1_steps, l1_hist = hist
    plot_results      (model, problem, x_test, u_exact, stage_index=0, n_sub=1,  save_dir=save_dir)
    plot_loss_history (loss_hist, l1_steps, l1_hist, stage_index=0, n_sub=1,    save_dir=save_dir)
    plot_colloc_points(colloc_cur, domain, "stage0_final", save_dir)

    results[1] = {"rel_l1_error": float(best_l1)}
    current_model = model
    current_n_sub = 1

    # ======================  后续层次循环  ==================================
    pou_schedule      = config.get("pou_schedule", [4, 9, 16, 25])
    discovery_pointer = 0
    stage_counter     = 1          # 从 1 开始给多域 FBPINN 计数

    while discovery_pointer < len(pou_schedule):

        # ------- PoU 发现 ---------------------------------------------------
        found_new = False
        for i in range(discovery_pointer, len(pou_schedule)):
            n_sub_try = pou_schedule[i]
            nx = int(np.sqrt(n_sub_try)); ny = n_sub_try // nx
            if nx * ny != n_sub_try:
                continue

            key, pou_key = jax.random.split(key)
            pou_net   = SepMLPPOUNet(nx, ny, key=pou_key, **config.get("sep_mlp_pou_conf", {}))
            init_pars = pou_net.init_params()

            x_pou = jnp.asarray(colloc_cur)                     # ← RAD 点
            y_pou = jax.vmap(current_model)(x_pou).squeeze()
            fin_pars = run_lsgd(pou_net, init_pars, x_pou, y_pou, config["lsgd_conf"])
            window_fn = WindowModule(pou_net=pou_net, params=fin_pars)

            if check_territory(window_fn, x_pou):
                # 画 PoU 权重图
                plot_pou_results(window_fn, problem, x_test,
                                 stage_index=stage_counter,
                                 n_sub_next=n_sub_try,
                                 save_dir=save_dir)
                best_n_sub   = n_sub_try
                best_window  = window_fn
                found_new    = True
            else:
                discovery_pointer = i
                break

        if not found_new:
            break

        # ------- 多域 FBPINN 训练 ------------------------------------------
        key, subkey = jax.random.split(key)
        model = FBPINN_PoU(
            key=subkey,
            num_subdomains=best_n_sub,
            mlp_config=config["mlp_conf"],
            window_fn=best_window,
            domain=domain,
            ansatz=problem.ansatz,
            residual_fn=problem.residual,
        )

        model, best_l1, hist, colloc_cur = train_fbpinn_no_batching(
            subkey, model, problem, colloc_cur,
            config["FBPINN_LR"], config["FBPINN_STEPS"],
            x_test, u_exact,
            rad_config=config.get("rad_config"),
        )

        # —— 画图
        loss_hist, l1_steps, l1_hist = hist
        plot_results      (model, problem, x_test, u_exact, stage_counter, best_n_sub, save_dir)
        plot_loss_history (loss_hist, l1_steps, l1_hist,  stage_counter, best_n_sub, save_dir)
        plot_colloc_points(colloc_cur, domain, f"stage{stage_counter}_final", save_dir)

        results[best_n_sub] = {"rel_l1_error": float(best_l1)}
        current_model = model
        current_n_sub = best_n_sub
        stage_counter += 1
        if best_n_sub == pou_schedule[discovery_pointer]:
            discovery_pointer += 1         # 继续探索更细的分区

    return results



# ==============================================================================
# 8. Script Execution Entrypoint
# ==============================================================================
if __name__ == '__main__':
    config = {
        "BATCH_SIZE": 2048, 
        "FBPINN_STEPS": 2000, "FBPINN_LR": 1e-3,
        "test_n_2d": 100, "colloc_n_2d": 100, 
        "mlp_conf": dict(in_size=2, out_size=1, width_size=16, depth=2, activation=jnp.tanh),
        "lsgd_conf": LSGDConfig(n_epochs=1000, lr=1e-4, n_stag=300),
        "sep_mlp_pou_conf": dict(hidden=(16, 16), tau=1.0),
        "pou_schedule": [4, 6, 9],
        
        # --- RAD Configuration ---
        # Note: The `n_draw` parameter is now dynamically set inside the training loops
        # to match the total number of collocation points, enabling 100% resampling.
        "rad_config": {
            "resample_every": 5000,  # Resample every 5000 steps
            "plot_colloc": True,      # Plot the collocation points after each resampling
            "sample_params": {
                "n_draw": 2000,       # This value is a placeholder; it's overridden in the loop
                "pool_size": 20000,   # Candidate pool size for sampling
                "k": 3.0,             # Residual weighting exponent
                "c": 1.0,             # Add a small constant to ensure non-zero probability
            }
        }
    }
    
    problem = ViscousBurgersFBPINN()
    if "rad_config" in config:
        config["rad_config"]["sample_params"]["domain"] = problem.domain

    key = jax.random.PRNGKey(42)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config["save_dir"] = f"results_{problem.__class__.__name__}_{timestamp}_RAD_Full"
    os.makedirs(config["save_dir"], exist_ok=True)
    print(f"Results will be saved to: {config['save_dir']}")

    final_results = run_hierarchical_fbpinn(key, problem, config)

    print("\n\n" + "#"*80); print("##### Algorithm execution finished #####"); print("#"*80)
    print(f"All results and plots have been saved to: '{config['save_dir']}'")
    print("\nSummary of Relative L1 Error for each stage:")
    sorted_results = sorted(final_results.items())
    for n_sub_val, res in sorted_results:
        model_type = "Base PINN (Batched)" if n_sub_val == 1 else "FBPINN (Full-batch)"
        print(f"   Model ({model_type}, n_sub={n_sub_val:<2}) | Relative L1 Error = {res['rel_l1_error']:.4e}")






