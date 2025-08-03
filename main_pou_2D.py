# %matplotlib widget
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
from physics.problems import Poisson2D_freq68
from model.fbpinn_model import FBPINN_PoU
from utils.data_utils import generate_collocation


# ==============================================================================
# 2. PoU学习相关模块
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
# 3. 局部拟合与PoU训练
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
    n_epochs: int = 15000; lr: float = 1e-4; lam_init: float = 0.01; rho: float = 0.99; n_stag: int = 200

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
# 4. 两个独立的训练函数
# ==============================================================================
def train_pinn_with_batching(key, model, problem, colloc, lr, steps, batch_size, x_test, u_exact, eval_every=100):
    params, static = eqx.partition(model, eqx.is_array)
    opt = optax.adam(lr); opt_state = opt.init(params)
    @eqx.filter_jit
    def loss_fn_wrapper(p, xy): return problem.residual(eqx.combine(p, static), xy)
    @eqx.filter_jit
    def step_fn(p, o, batch_colloc):
        loss, g = jax.value_and_grad(loss_fn_wrapper)(p, batch_colloc)
        updates, o = opt.update(g, o); p = eqx.apply_updates(p, updates); return p, o, loss
    @eqx.filter_jit
    def eval_fn(p):
        m = eqx.combine(p, static); pred = jax.vmap(m)(x_test).squeeze(); u_exact_squeezed = u_exact.squeeze()
        return jnp.mean(jnp.abs(pred - u_exact_squeezed))
    print("JIT compiling PINN trainer (with batching)...", end="", flush=True)
    step_fn(params, opt_state, colloc[:batch_size]); print(" done")
    loss_hist, l1_hist, l1_steps, best_l1, best_params = [], [], [], np.inf, params
    bar = trange(steps, desc="PINN (N=1, batched)", dynamic_ncols=True)
    for s in bar:
        key, step_key = jax.random.split(key)
        batch_indices = jax.random.choice(step_key, a=len(colloc), shape=(batch_size,), replace=False)
        params, opt_state, loss = step_fn(params, opt_state, colloc[batch_indices])
        loss_hist.append(float(loss))
        if (s + 1) % eval_every == 0 or s + 1 == steps:
            l1_error = float(eval_fn(params)); l1_hist.append(l1_error); l1_steps.append(s + 1)
            bar.set_postfix(loss=f"{loss:.3e}", L1_err=f"{l1_error:.3e}")
            if l1_error < best_l1: best_l1 = l1_error; best_params = params
        else: bar.set_postfix(loss=f"{loss:.3e}")
    final_model = eqx.combine(best_params, static)
    return final_model, best_l1, (jnp.array(loss_hist), jnp.array(l1_steps), jnp.array(l1_hist))

def train_fbpinn_no_batching(key, model, problem, colloc, lr, steps, x_test, u_exact, eval_every=100):
    params, static = eqx.partition(model, eqx.is_array)
    opt = optax.adam(lr); opt_state = opt.init(params)
    @eqx.filter_jit
    def loss_fn_wrapper(p, xy): return problem.residual(eqx.combine(p, static), xy)
    @eqx.filter_jit
    def step_fn(p, o, all_colloc):
        loss, g = jax.value_and_grad(loss_fn_wrapper)(p, all_colloc)
        updates, o = opt.update(g, o); p = eqx.apply_updates(p, updates); return p, o, loss
    @eqx.filter_jit
    def eval_fn(p):
        m = eqx.combine(p, static); pred = jax.vmap(m)(x_test).squeeze(); u_exact_squeezed = u_exact.squeeze()
        return jnp.mean(jnp.abs(pred - u_exact_squeezed))
    print("JIT compiling FBPINN trainer (full-batch)...", end="", flush=True)
    step_fn(params, opt_state, colloc); print(" done")
    loss_hist, l1_hist, l1_steps, best_l1, best_params = [], [], [], np.inf, params
    bar = trange(steps, desc=f"FBPINN (N={len(model.subnets)}, full-batch)", dynamic_ncols=True)
    for s in bar:
        params, opt_state, loss = step_fn(params, opt_state, colloc)
        loss_hist.append(float(loss))
        if (s + 1) % eval_every == 0 or s + 1 == steps:
            l1_error = float(eval_fn(params)); l1_hist.append(l1_error); l1_steps.append(s + 1)
            bar.set_postfix(loss=f"{loss:.3e}", L1_err=f"{l1_error:.3e}")
            if l1_error < best_l1: best_l1 = l1_error; best_params = params
        else: bar.set_postfix(loss=f"{loss:.3e}")
    final_model = eqx.combine(best_params, static)
    return final_model, best_l1, (jnp.array(loss_hist), jnp.array(l1_steps), jnp.array(l1_hist))

# ==============================================================================
# 5. 分析函数
# ==============================================================================
def check_territory(window_fn: Callable, x_test: jax.Array, dominance_threshold: float = 0.5) -> bool:
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
    u_pred = jax.vmap(model)(x_test); u_pred = u_pred.reshape(-1)
    u_ex = u_exact.reshape(-1)
    if u_pred.shape != u_ex.shape:
        print(f"Warning: Shape mismatch! u_pred: {u_pred.shape}, u_ex: {u_ex.shape}")
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
    print(f"         ... plots saved to {filepath}")

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
    print(f"         ... plots saved to {filepath}")

def plot_loss_history(loss_hist, l1_steps, l1_hist, stage_index, n_sub, save_dir):
    print(f"   -> Stage {stage_index}: Generating loss history plot (n_sub={n_sub})...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6)); ax.plot(loss_hist, label='Training Loss (MSE)', color='tab:blue', alpha=0.8); ax.set_yscale('log'); ax.set_xlabel('Steps'); ax.set_ylabel('Loss', color='tab:blue'); ax.grid(True, which="both", ls="--", alpha=0.5)
    ax2 = ax.twinx(); ax2.plot(l1_steps, l1_hist, 'o-', label='Evaluation Metric (L1 Error)', color='tab:red'); ax2.set_ylabel('Relative L1 Error', color='tab:red')
    fig.suptitle(f'Stage {stage_index}: History (n_sub={n_sub})', fontsize=16); lines, labels = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best'); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filepath = os.path.join(save_dir, f"loss_history_stage_{stage_index}_nsub_{n_sub}.png"); plt.savefig(filepath, dpi=300); plt.close(fig)
    print(f"         ... plot saved to {filepath}")

def save_npz_history(save_dir, stage_index, n_sub, history_data):
    loss_hist, l1_steps, l1_hist = history_data
    filename = f"history_stage_{stage_index}_nsub_{n_sub}.npz"
    filepath = os.path.join(save_dir, filename); print(f"   -> Saving training history to {filepath}...")
    np.savez_compressed(filepath, loss_hist=np.asarray(loss_hist), l1_steps=np.asarray(l1_steps), l1_hist=np.asarray(l1_hist))
    print(f"         ... history saved.")

# ==============================================================================
# 6. 主迭代逻辑
# ==============================================================================
def run_hierarchical_fbpinn(key, problem, config):
    domain = problem.domain; problem_dim = len(domain[0]); save_dir = config["save_dir"]
    if problem_dim == 1:
        raise NotImplementedError("1D case is not configured for this specific training logic.")
    
    test_n_2d = config.get("test_n_2d", 101); colloc_n_2d = config.get("colloc_n_2d", 100)
    x_test = generate_collocation(domain, test_n_2d, strategy="grid")
    colloc_train = generate_collocation(domain, colloc_n_2d, strategy="grid")
    print(f"Generated {colloc_train.shape[0]} collocation points for training.")
    u_exact = jax.vmap(problem.exact)(x_test).squeeze()
    results = {}
    
    # --- Stage 0: Base PINN (N=1) ---
    print("\n" + "="*80); print("===== Stage 0: Training Base PINN Model (n_sub=1) with BATCHING ====="); print("="*80)
    key, stage_key = jax.random.split(key)
    current_model = FBPINN_PoU(
        key=stage_key, num_subdomains=1, mlp_config=config["mlp_conf"],
        domain=problem.domain, ansatz=problem.ansatz, residual_fn=problem.residual
    )
    current_model, best_l1, history = train_pinn_with_batching(
        stage_key, current_model, problem, colloc_train, config["FBPINN_LR"],
        config["FBPINN_STEPS"], config["BATCH_SIZE"], x_test, u_exact
    )
    
    current_n_sub = 1; results[current_n_sub] = {'rel_l1_error': float(best_l1)}
    print(f"\nTraining for n_sub=1 complete. Final Relative L1 Error: {best_l1:.4e}")
    plot_results(current_model, problem, x_test, u_exact, 0, current_n_sub, save_dir, test_n_2d=test_n_2d)
    plot_loss_history(*history, 0, current_n_sub, save_dir); save_npz_history(save_dir, 0, current_n_sub, history)

    # --- 分层迭代主循环 (N>1) ---
    pou_discovery_schedule = config.get("pou_schedule", [4, 9, 16, 25])
    discovery_start_index = 0; outer_loop_iter = 1
    
    while True:
        print("\n" + "#"*80); print(f"##### Iteration {outer_loop_iter}: Discovering Partitions... #####"); print("#"*80)
        if discovery_start_index >= len(pou_discovery_schedule): print("\nAll schemes attempted. Converged."); break

        y_train_pou = jax.vmap(current_model)(x_test).squeeze()
        best_n_sub_in_phase = current_n_sub; best_window_fn_in_phase = getattr(current_model, 'window_fn', None)
        found_new_qualified_partition = False
        
        for i in range(discovery_start_index, len(pou_discovery_schedule)):
            n_sub_test = pou_discovery_schedule[i]; print(f"\n-- Testing health of {n_sub_test} partitions --")
            key, pou_key = jax.random.split(key)
            nx, ny = get_grid_dims(n_sub_test)
            if nx * ny != n_sub_test: print(f"   -> Skipping {n_sub_test}."); continue
            
            print(f"   -> Learning 2D SepMLP-PoU for {n_sub_test} = {nx}x{ny} partitions...")
            pou_net = SepMLPPOUNet(nx=nx, ny=ny, key=pou_key, **config.get("sep_mlp_pou_conf", {}))
            initial_params = pou_net.init_params()
            final_pou_params = run_lsgd(pou_net, initial_params, x_test, y_train_pou, config["lsgd_conf"])
            temp_window_fn = WindowModule(pou_net=pou_net, params=final_pou_params)
            
            if check_territory(temp_window_fn, x_test):
                print(f"   -> {n_sub_test} partitions scheme is healthy."); best_n_sub_in_phase = n_sub_test
                best_window_fn_in_phase = temp_window_fn; found_new_qualified_partition = True
                
                # 【已修正】移除 plot_pou_results 调用中多余的 u_exact 参数
                plot_pou_results(temp_window_fn, problem, x_test, outer_loop_iter, n_sub_test, save_dir, test_n_2d=test_n_2d)
            else:
                print(f"   -> Territory loss at {n_sub_test} partitions! Stopping complexity increase."); discovery_start_index = i; break
        else: discovery_start_index = len(pou_discovery_schedule)
        
        if not found_new_qualified_partition: print("\nNo new qualified partition found. Converged."); break
        
        # --- FBPINN (N>1) ---
        print("\n" + "="*80); print(f"===== Stage {outer_loop_iter}: FBPINN Training (n_sub={best_n_sub_in_phase}) with FULL BATCH ====="); print("="*80)
        key, stage_key = jax.random.split(key); current_n_sub = best_n_sub_in_phase
        current_model = FBPINN_PoU(
            key=stage_key, num_subdomains=current_n_sub, mlp_config=config["mlp_conf"], window_fn=best_window_fn_in_phase,
            domain=problem.domain, ansatz=problem.ansatz, residual_fn=problem.residual
        )
        current_model, best_l1, history = train_fbpinn_no_batching(
            stage_key, current_model, problem, colloc_train, config["FBPINN_LR"],
            config["FBPINN_STEPS"], x_test, u_exact
        )
        
        results[current_n_sub] = {'rel_l1_error': float(best_l1)}
        print(f"\nTraining for n_sub={current_n_sub} complete. Final Relative L1 Error: {best_l1:.4e}")
        plot_results(current_model, problem, x_test, u_exact, outer_loop_iter, current_n_sub, save_dir, test_n_2d=test_n_2d)
        plot_loss_history(*history, outer_loop_iter, current_n_sub, save_dir); save_npz_history(save_dir, outer_loop_iter, current_n_sub, history)
        
        outer_loop_iter += 1
    return results

# ==============================================================================
# 7. 脚本执行入口
# ==============================================================================
if __name__ == '__main__':
    config = {
        "BATCH_SIZE": 2048, 
        "FBPINN_STEPS": 30000, "FBPINN_LR": 1e-3,
        "test_n_2d": 101, "colloc_n_2d": 100, 
        "mlp_conf": dict(in_size=2, out_size=1, width_size=64, depth=3, activation=jnp.tanh),
        "lsgd_conf": LSGDConfig(n_epochs=5000, lr=1e-4, n_stag=300),
        "sep_mlp_pou_conf": dict(hidden=(16, 16), tau=3),
        "pou_schedule": [4, 8, 16] 
    }
    
    key = jax.random.PRNGKey(42); problem = Poisson2D_freq68() 
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config["save_dir"] = f"results_{problem.__class__.__name__}_{timestamp}"
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