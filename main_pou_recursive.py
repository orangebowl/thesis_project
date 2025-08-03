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
from physics.problems import SineX3ODE
from model.fbpinn_model import FBPINN_PoU
from utils.data_utils import generate_collocation


# ==============================================================================
# 2. PoU学习相关模块
# ==============================================================================
class RBFPOUNet:
    """基于RBF的PoU网络，用于1D问题"""
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
        d2 = jnp.sum((x[:, None, :] - c[None, :, :])**2, -1)
        log_phi = -d2 / (w**2 + 1e-12)
        log_phi = log_phi - jnp.max(log_phi, 1, keepdims=True)
        phi = jnp.exp(log_phi)
        phi = jnp.maximum(phi, 1e-8)
        return phi / jnp.sum(phi, 1, keepdims=True)

class WindowModule(eqx.Module):
    pou_net: Any = eqx.static_field()
    params: Dict
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.pou_net.forward(self.params, x)

# ==============================================================================
# 3. 局部拟合与PoU训练
# ==============================================================================
def _design_matrix(x: jnp.ndarray) -> jnp.ndarray:
    d = x.shape[-1]
    if d == 1:
        x_flat = x.squeeze()
        return jnp.stack([jnp.ones_like(x_flat), x_flat, x_flat**2], -1)
    elif d == 2:
        x1, x2 = x[:, 0], x[:, 1]
        return jnp.stack([jnp.ones_like(x1), x1, x2, x1**2, x1*x2, x2**2], -1)
    else:
        raise ValueError("Only 1-D or 2-D supported")

def fit_local_polynomials(x, y, w, lam: float = 0.0):
    A, y = _design_matrix(x), y[:, None]; k = A.shape[-1]
    def _solve(weights):
        Aw = A * weights[:, None]; M  = A.T @ Aw; b  = (Aw.T @ y).squeeze(-1)
        return jnp.linalg.solve(M + lam*jnp.eye(k), b)
    return jax.vmap(_solve, 1, 0)(w)

def _predict_from_coeffs(x, coeffs, partitions):
    A = _design_matrix(x); y_cent = A @ coeffs.T
    return jnp.sum(partitions * y_cent, 1)

@dataclasses.dataclass
class LSGDConfig:
    n_epochs: int = 15000; lr: float = 1e-4; lam_init: float = 0.01; rho: float = 0.99; n_stag: int = 100

def run_lsgd(pou_net, initial_params, x, y, cfg: LSGDConfig):
    params = initial_params
    def loss_fn(p, lam):
        part = pou_net.forward(p, x)
        coeffs = fit_local_polynomials(x, y, part, lam)
        pred = _predict_from_coeffs(x, coeffs, part)
        return jnp.mean((pred - y)**2)
    valgrad_fn = jax.jit(jax.value_and_grad(loss_fn))
    opt = optax.adam(cfg.lr); opt_state = opt.init(params)
    lam = jnp.array(cfg.lam_init); best, stag = jnp.inf, 0
    bar = trange(cfg.n_epochs, desc=f"PoU-LSGD (N={pou_net.num_experts})", dynamic_ncols=True)
    for ep in bar:
        loss_val, grads = valgrad_fn(params, lam)
        updates, opt_state = opt.update(grads, opt_state)
        params = eqx.apply_updates(params, updates)
        bar.set_postfix(loss=f"{loss_val:.3e}")
        if loss_val < best - 1e-12: best, stag = loss_val, 0
        else: stag += 1
        if stag > cfg.n_stag: lam *= cfg.rho; stag = 0; print(f"\nStagnation. Lambda decayed to {lam:.4f}")
    return params

# ==============================================================================
# 4. FBPINN训练与分析函数
# ==============================================================================
def train_fbpinn_simple(key, model, problem, colloc, lr, steps, x_test, u_exact, eval_every=100):
    params, static = eqx.partition(model, eqx.is_array); opt = optax.adam(lr); opt_state = opt.init(params)
    
    @eqx.filter_jit
    def loss_fn_wrapper(p, xy): return problem.residual(eqx.combine(p, static), xy)
    
    @eqx.filter_jit
    def step_fn(p, o, xb):
        loss, g = jax.value_and_grad(loss_fn_wrapper)(p, xb)
        updates, o = opt.update(g, o); p = eqx.apply_updates(p, updates); return p, o, loss
        
    @eqx.filter_jit
    def eval_fn(p):
        m = eqx.combine(p, static)
        pred = jax.vmap(m)(x_test).squeeze()
        return jnp.mean(jnp.abs(pred - u_exact.squeeze()))

    print("JIT compiling FBPINN trainer...", end="", flush=True); step_fn(params, opt_state, colloc[:1]); print(" done")
    loss_hist, l1_hist, l1_steps, best_l1, best_params = [], [], [], np.inf, params
    bar = trange(steps, desc=f"FBPINN (N={len(model.subnets)})", dynamic_ncols=True)
    for s in bar:
        params, opt_state, loss = step_fn(params, opt_state, colloc)
        loss_hist.append(float(loss))
        if (s + 1) % eval_every == 0 or s + 1 == steps:
            l1_error = float(eval_fn(params)); l1_hist.append(l1_error); l1_steps.append(s + 1)
            bar.set_postfix(loss=f"{loss:.3e}", L1_err=f"{l1_error:.3e}")
            if l1_error < best_l1:
                best_l1 = l1_error
                best_params = params
        else:
            bar.set_postfix(loss=f"{loss:.3e}")
            
    final_model = eqx.combine(best_params, static)
    return final_model, best_l1, (jnp.array(loss_hist), jnp.array(l1_steps), jnp.array(l1_hist))

def check_territory(window_fn: Callable, x_test: jax.Array, dominance_threshold: float = 0.5) -> bool:
    weights = window_fn(x_test); num_partitions = weights.shape[1]
    if num_partitions == 1: return True
    dominant_partition_indices = jnp.argmax(weights, axis=1)
    unique_dominant_indices = jnp.unique(dominant_partition_indices)
    if len(unique_dominant_indices) < num_partitions:
        missing_indices = jnp.setdiff1d(jnp.arange(num_partitions), unique_dominant_indices)
        print(f"      - HEALTH CHECK FAILED: Partitions {missing_indices} have no territory!")
        return False
    for i in range(num_partitions):
        weights_in_territory = weights[dominant_partition_indices == i, i]
        if weights_in_territory.size == 0: continue
        max_weight_in_territory = jnp.max(weights_in_territory)
        if max_weight_in_territory < dominance_threshold:
            print(f"      - HEALTH CHECK FAILED: Partition {i} is weak (peak weight {max_weight_in_territory:.3e} < {dominance_threshold}).")
            return False
    print(f"      - Health check passed for {num_partitions} partitions.")
    return True

def get_grid_dims(n_sub: int) -> Tuple[int, int]:
    if n_sub == 1: return 1, 1
    if n_sub < 4: return n_sub, 1
    best_factor = 1
    for i in range(2, int(jnp.sqrt(n_sub)) + 1):
        if n_sub % i == 0: best_factor = i
    return (n_sub // best_factor, best_factor)

def plot_results(model, problem, x_test, stage_index, n_sub, save_dir, test_n_2d=100):
    problem_dim = len(problem.domain[0])
    print(f"  -> Stage {stage_index}: Generating FBPINN solution plots (n_sub={n_sub})...")
    if problem_dim == 1:
        u_pred = jax.vmap(model)(x_test).squeeze(); u_ex = problem.exact(x_test).squeeze()
        fig, ax = plt.subplots(1, 1, figsize=(10, 6)); ax.plot(x_test, u_ex, 'k-', label='Exact', linewidth=2); ax.plot(x_test, u_pred, 'r--', label='Predicted')
        ax.set_title(f'Stage {stage_index}: FBPINN Result ({n_sub} subdomains)'); ax.set_xlabel('x'); ax.set_ylabel('u(x)'); ax.legend(); ax.grid(True)
        filepath = os.path.join(save_dir, f"fbpinn_stage_{stage_index}_nsub_{n_sub}.png")
    else: # 2D
        u_pred = jax.vmap(model)(x_test).squeeze(); u_ex = problem.exact(x_test).squeeze(); error = jnp.abs(u_pred - u_ex)
        u_pred_grid = u_pred.reshape(test_n_2d, test_n_2d); u_ex_grid = u_ex.reshape(test_n_2d, test_n_2d); error_grid = error.reshape(test_n_2d, test_n_2d); fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        domain_extent = [problem.domain[0][0], problem.domain[1][0], problem.domain[0][1], problem.domain[1][1]]
        im1 = axes[0].imshow(u_pred_grid, extent=domain_extent, origin='lower', cmap='viridis'); axes[0].set_title('Predicted'); fig.colorbar(im1, ax=axes[0])
        im2 = axes[1].imshow(u_ex_grid, extent=domain_extent, origin='lower', cmap='viridis'); axes[1].set_title('Exact'); fig.colorbar(im2, ax=axes[1])
        im3 = axes[2].imshow(error_grid, extent=domain_extent, origin='lower', cmap='Reds'); axes[2].set_title('Error'); fig.colorbar(im3, ax=axes[2])
        fig.suptitle(f'Stage {stage_index}: FBPINN Result ({n_sub} subdomains)', fontsize=16); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filepath = os.path.join(save_dir, f"fbpinn_stage_{stage_index}_nsub_{n_sub}.png")
    plt.savefig(filepath, dpi=300); plt.close(fig)
    print(f"     ... plots saved to {filepath}")

def plot_pou_results(window_fn, problem, x_test, stage_index, n_sub_next, save_dir, test_n_2d=100):
    problem_dim = len(problem.domain[0])
    print(f"  -> Stage {stage_index}: Generating PoU partition plots for next stage (n_sub={n_sub_next})...")
    weights = window_fn(x_test); num_windows = weights.shape[1]
    if problem_dim == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for i in range(weights.shape[1]): ax.plot(x_test, weights[:, i], label=f'Window {i+1}')
        ax.set_title(f'Stage {stage_index}: Learned PoU for {n_sub_next} Subdomains'); ax.set_xlabel('x'); ax.set_ylabel('Weight'); ax.legend(); ax.grid(True); ax.set_ylim(-0.1, 1.1)
        filepath = os.path.join(save_dir, f"pou_stage_{stage_index}_nsub_next_{n_sub_next}.png")
    else: # 2D
        nx, ny = get_grid_dims(num_windows)
        fig, axes = plt.subplots(ny, nx, figsize=(4 * nx, 3.5 * ny), squeeze=False); axes = axes.ravel(); domain_extent = [problem.domain[0][0], problem.domain[1][0], problem.domain[0][1], problem.domain[1][1]]
        for i in range(num_windows):
            ax = axes[i]; window_grid = weights[:, i].reshape(test_n_2d, test_n_2d); im = ax.imshow(window_grid, extent=domain_extent, origin='lower', cmap='inferno', vmin=0, vmax=1)
            ax.set_title(f'Window {i+1}'); fig.colorbar(im, ax=ax)
        for j in range(num_windows, len(axes)): axes[j].axis('off')
        fig.suptitle(f'Stage {stage_index}: Learned PoU for {n_sub_next} Subdomains', fontsize=16); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filepath = os.path.join(save_dir, f"pou_stage_{stage_index}_nsub_next_{n_sub_next}.png")
    plt.savefig(filepath, dpi=300); plt.close(fig)
    print(f"     ... plots saved to {filepath}")

def plot_loss_history(loss_hist, l1_steps, l1_hist, stage_index, n_sub, save_dir):
    print(f"  -> Stage {stage_index}: Generating loss history plot (n_sub={n_sub})...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(loss_hist, label='Training Loss', color='tab:blue', alpha=0.8); ax.set_yscale('log'); ax.set_xlabel('Steps'); ax.set_ylabel('Loss', color='tab:blue'); ax.grid(True, which="both", ls="--", alpha=0.5)
    ax2 = ax.twinx(); ax2.plot(l1_steps, l1_hist, 'o-', label='L1 Error', color='tab:red'); ax2.set_ylabel('L1 Error', color='tab:red')
    fig.suptitle(f'Stage {stage_index}: History (n_sub={n_sub})', fontsize=16); fig.legend(loc="upper right"); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filepath = os.path.join(save_dir, f"loss_history_stage_{stage_index}_nsub_{n_sub}.png"); plt.savefig(filepath, dpi=300); plt.close(fig)
    print(f"     ... plot saved to {filepath}")

# --- 新增: 保存训练历史到NPZ文件 ---
def save_npz_history(save_dir, stage_index, n_sub, history_data):
    """Saves training history to a compressed .npz file."""
    loss_hist, l1_steps, l1_hist = history_data
    filename = f"history_stage_{stage_index}_nsub_{n_sub}.npz"
    filepath = os.path.join(save_dir, filename)
    
    print(f"  -> Saving training history to {filepath}...")
    np.savez_compressed(
        filepath,
        loss_hist=np.asarray(loss_hist),
        l1_steps=np.asarray(l1_steps),
        l1_hist=np.asarray(l1_hist)
    )
    print(f"     ... history saved.")

# ==============================================================================
# 5. 主迭代逻辑 - 重构为函数
# ==============================================================================
def run_hierarchical_fbpinn(key, problem, config):
    """
    执行完整的分层自适应FBPINN算法。
    """
    # --- 配置和数据准备 ---
    domain = problem.domain
    problem_dim = len(domain[0])
    save_dir = config["save_dir"]
    
    if problem_dim == 1:
        x_test = jnp.linspace(domain[0][0], domain[1][0], config["test_n"])[:, None]
        u_exact_fn = jax.vmap(problem.exact)
        u_exact = u_exact_fn(x_test)
        colloc_train = generate_collocation(domain, config["colloc_n"])
    else: # 2D case
        test_n_2d = config.get("test_n_2d", 100)
        colloc_n_2d = config.get("colloc_n_2d", 60)
        x_test = generate_collocation(domain, test_n_2d**2)
        u_exact = jax.vmap(problem.exact)(x_test)
        colloc_train = generate_collocation(domain, colloc_n_2d**2, method="sobol")

    results = {}
    
    # --- 算法初始化: 训练基础 PINN 模型 (n_sub=1) ---
    print("\n" + "="*80); print("===== 算法初始化: 训练基础 PINN 模型 (n_sub=1) ====="); print("="*80)
    key, pinn_key = jax.random.split(key)
    current_model = FBPINN_PoU(
        key=pinn_key, domain=domain, num_subdomains=1, mlp_config=config["mlp_conf"],
        ansatz=problem.ansatz, residual_fn=problem.residual,
        window_fn=None, window_on_physical=True
    )
    current_model, best_l1, history = train_fbpinn_simple(
        key, current_model, problem, colloc_train, config["FBPINN_LR"],
        config["FBPINN_STEPS"], x_test, u_exact
    )
    current_n_sub = 1
    results[current_n_sub] = {'l1_error': float(best_l1)}
    print(f"\n基础 PINN 训练完成. L1 Error: {best_l1:.4e}")
    plot_results(current_model, problem, x_test, 0, current_n_sub, save_dir, test_n_2d=locals().get('test_n_2d',100))
    plot_loss_history(*history, 0, current_n_sub, save_dir)
    save_npz_history(save_dir, 0, current_n_sub, history) # <-- 新增：保存历史数据

    # --- 分层迭代主循环 ---
    pou_discovery_schedule = [4, 8, 16, 20, 30]
    discovery_start_index = 0
    outer_loop_iter = 1
    
    while True:
        print("\n" + "#"*80); print(f"##### 开始第 {outer_loop_iter} 轮分层迭代: 基于 n_sub={current_n_sub} 的解进行探索 #####"); print("#"*80)
        
        # 关键Bug修复：在循环开始时检查是否已探索完所有方案
        if discovery_start_index >= len(pou_discovery_schedule):
            print("\n所有预定的分区方案均已成功探索或失败，没有更复杂的方案可供测试。算法收敛。")
            break

        y_train_pou = jax.vmap(current_model)(colloc_train).squeeze()
        best_n_sub_in_phase = current_n_sub
        best_window_fn_in_phase = getattr(current_model, 'window_fn', None)
        found_new_qualified_partition = False
        print(f"本轮 PoU 探索将从索引 {discovery_start_index} ({pou_discovery_schedule[discovery_start_index]}个分区) 开始。")
        
        # --- 内循环: PoU 分区发现 ---
        for i in range(discovery_start_index, len(pou_discovery_schedule)):
            n_sub_test = pou_discovery_schedule[i]
            print(f"\n-- 正在测试 {n_sub_test} 个分区方案的健康状况 --")
            key, pou_key = jax.random.split(key)
            
            if problem_dim == 1:
                pou_net = RBFPOUNet(input_dim=problem_dim, num_centers=n_sub_test, domain=domain, key=pou_key)
                initial_params = pou_net.init_params()
                final_pou_params = run_lsgd(pou_net, initial_params, colloc_train, y_train_pou, config["lsgd_conf"])
                temp_window_fn = WindowModule(pou_net=pou_net, params=final_pou_params)
            else:
                print("2D PoU 学习尚未实现，跳过...")
                break

            if check_territory(temp_window_fn, x_test):
                print(f"  -> {n_sub_test} 个分区的方案健康。这是一个合格的候选分区。")
                best_n_sub_in_phase = n_sub_test
                best_window_fn_in_phase = temp_window_fn
                found_new_qualified_partition = True
                plot_pou_results(temp_window_fn, problem, x_test, outer_loop_iter, n_sub_test, save_dir, test_n_2d=locals().get('test_n_2d',100))
            else:
                print(f"  -> 在 {n_sub_test} 个分区处检测到领地丢失！停止在该轮次增加分区复杂度。")
                discovery_start_index = i
                break
        else:
            # 仅当for循环正常结束时执行
            print(f"\n-- PoU探索完成，所有候选分区均健康。--")
            discovery_start_index = len(pou_discovery_schedule)
        
        if not found_new_qualified_partition:
            print("\n在本轮探索中未能找到任何新的合格分区方案，算法收敛。")
            break
        
        # --- 外循环: 训练更高层级的 FBPINN ---
        print("\n" + "="*80); print(f"===== 第 {outer_loop_iter} 轮 FBPINN 训练 (n_sub = {best_n_sub_in_phase}) ====="); print("="*80)
        current_n_sub = best_n_sub_in_phase
        key, fbpinn_key = jax.random.split(key)
        
        current_model = FBPINN_PoU(
            key=fbpinn_key, domain=domain, num_subdomains=current_n_sub, mlp_config=config["mlp_conf"],
            ansatz=problem.ansatz, residual_fn=problem.residual,
            window_fn=best_window_fn_in_phase, window_on_physical=True
        )
        current_model, best_l1, history = train_fbpinn_simple(
            key, current_model, problem, colloc_train, config["FBPINN_LR"],
            config["FBPINN_STEPS"], x_test, u_exact
        )
        
        results[current_n_sub] = {'l1_error': float(best_l1)}
        print(f"\nFBPINN (n_sub={current_n_sub}) 训练完成. L1 Error: {best_l1:.4e}")
        plot_results(current_model, problem, x_test, outer_loop_iter, current_n_sub, save_dir, test_n_2d=locals().get('test_n_2d',100))
        plot_loss_history(*history, outer_loop_iter, current_n_sub, save_dir)
        save_npz_history(save_dir, outer_loop_iter, current_n_sub, history) # <-- 新增：保存历史数据
        
        outer_loop_iter += 1
        
    return results

# ==============================================================================
# 6. 脚本执行入口
# ==============================================================================
if __name__ == '__main__':
    # --- 全局配置 ---
    config = {
        "FBPINN_STEPS": 30000,
        "FBPINN_LR": 1e-3,
        "test_n": 2000,
        "colloc_n": 3000,
        "test_n_2d": 100, # for 2D plotting
        "colloc_n_2d": 60, # for 2D training
        "mlp_conf": dict(in_size=1, out_size=1, width_size=32, depth=3, activation=jax.nn.tanh),
        "lsgd_conf": LSGDConfig(n_epochs=15000)
    }
    
    # --- 初始化和执行 ---
    key = jax.random.PRNGKey(42)
    problem = SineX3ODE()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config["save_dir"] = f"results_{problem.__class__.__name__}_{timestamp}"
    os.makedirs(config["save_dir"], exist_ok=True)
    print(f"结果将保存至: {config['save_dir']}")

    final_results = run_hierarchical_fbpinn(key, problem, config)

    # --- 最终总结 ---
    print("\n\n" + "#"*80); print("##### 算法执行完毕 #####"); print("#"*80)
    print(f"所有结果和图表已保存至: '{config['save_dir']}'")
    print("\n各层级模型 L1 误差总结:")
    sorted_results = sorted(final_results.items())
    for n_sub_val, res in sorted_results:
        model_type = "基础 PINN" if n_sub_val == 1 else "FBPINN"
        print(f"  模型 ({model_type}, n_sub={n_sub_val:<2}) | L1 Error = {res['l1_error']:.4e}")