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

from physics.problems import SineX3ODE,LocalHighFreqPoisson1D
from model.fbpinn_model import FBPINN_PoU 
from utils.data_utils import generate_collocation

# PoU 
class RBFPOUNet:
    """基于RBF的PoU网络，用于1D问题"""
    def __init__(self, input_dim: int, num_centers: int, domain, key=None):
        self.input_dim, self.num_experts = input_dim, num_centers
        key = jax.random.PRNGKey(42) if key is None else key
        min_b, max_b = jnp.array(domain[0]), jnp.array(domain[1])
        # 在实际定义域内均匀初始化中心点，增加少量扰动
        base_centers = jnp.linspace(min_b, max_b, num_centers)
        jitters = 0.05 * (max_b - min_b) * jax.random.normal(key, base_centers.shape)
        self._init_centers = base_centers + jitters
        # 宽度也应与定义域范围相关
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

        # 确保权重有一个最小阈值，防止过小的权重
        phi = jnp.maximum(phi, 1e-6)  # 设置一个最小的权重值
        return phi / jnp.sum(phi, 1, keepdims=True)

class WindowModule(eqx.Module):
    """一个Equinox模块，用于封装训练好的PoU网络，使其能作为FBPINN的窗函数"""
    pou_net: Any; params: Optional[Dict] = eqx.static_field(default=None)
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.pou_net.forward(self.params, x)

# ==============================================================================
# 3. 局部拟合与PoU训练
# ==============================================================================
def _design_matrix(x: jnp.ndarray) -> jnp.ndarray:
    d = x.shape[-1]
    if d == 1:
        # 新增 x.squeeze()**2 来包含二次项
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
    n_epochs: int = 5000; lr: float = 0.05; lam_init: float = 0.1; rho: float = 0.99; n_stag: int = 100

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
    print("PoU trainer setup complete. Starting training...")
    bar = trange(cfg.n_epochs, desc="PoU-LSGD", dynamic_ncols=True)
    for ep in bar:
        loss_val, grads = valgrad_fn(params, lam)
        updates, opt_state = opt.update(grads, opt_state)
        params = eqx.apply_updates(params, updates)
        bar.set_postfix(loss=f"{loss_val:.3e}")
        if loss_val < best - 1e-12: best, stag = loss_val, 0
        else: stag += 1
        if stag > cfg.n_stag: lam *= cfg.rho; stag = 0; print(f"\nStagnation. Lambda decayed to {lam:.4f}")
    return params

#  FBPINN训练与分析函数
def train_fbpinn_simple(key, model, problem, colloc, lr, steps, x_test, u_exact, eval_every=100):
    params, static = eqx.partition(model, eqx.is_array); opt = optax.adam(lr); opt_state = opt.init(params)
    def loss_fn_wrapper(p, xy): return problem.residual(eqx.combine(p, static), xy)
    @eqx.filter_jit
    def step_fn(p, o, xb):
        loss, g = jax.value_and_grad(loss_fn_wrapper)(p, xb)
        updates, o = opt.update(g, o); p = eqx.apply_updates(p, updates); return p, o, loss
    @eqx.filter_jit
    def eval_fn(p):
        m = eqx.combine(p, static)
        pred = jax.vmap(m)(x_test).squeeze()
        exact = problem.exact(x_test).squeeze(); return jnp.mean(jnp.abs(pred - exact))
    print("JIT compiling FBPINN trainer...", end="", flush=True); step_fn(params, opt_state, colloc[:1]); print(" done")
    loss_hist, l1_hist, l1_steps = [], [], []
    bar = trange(steps, desc=f"FBPINN (n_sub={len(model.subnets)})", dynamic_ncols=True)
    for s in bar:
        params, opt_state, loss = step_fn(params, opt_state, colloc); loss_hist.append(float(loss))
        if (s + 1) % eval_every == 0 or s + 1 == steps:
            l1_error = float(eval_fn(params)); l1_hist.append(l1_error); l1_steps.append(s + 1)
            bar.set_postfix(loss=f"{loss:.3e}", L1_err=f"{l1_error:.3e}")
        else: bar.set_postfix(loss=f"{loss:.3e}")
    return eqx.combine(params, static), (jnp.array(loss_hist), jnp.array(l1_steps), jnp.array(l1_hist))

def check_territory(window_fn, x_test, *, thr: float = 0.1) -> bool:
    """
    判定 PoU 窗口是否 collapse：
    只要每个分区在测试网格上的最大权重大于 `thr`
    （默认 0.1），就视作拥有领地，返回 True，否则 False。
    """
    weights = window_fn(x_test)                 # (N, n_sub)
    max_w    = jnp.max(weights, axis=0)         # (n_sub,)

    # 找出最大权重 ≤ thr 的分区
    collapsed_mask = max_w <= thr
    if jnp.any(collapsed_mask):
        bad_idx = jnp.where(collapsed_mask)[0]
        print(f"    - 警告: 分区 {bad_idx} 最大权重 ≤ {thr}, 发生 collapse！")
        return False

    print(f"    - Info: 所有分区最大权重 > {thr}，均有领地。")
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
    print(f"  -> Stage {stage_index}: Generating FBPINN solution plots (n_sub={n_sub})...")
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
    print(f"     ... plots saved to {filepath}")

def plot_pou_results(window_fn, problem, x_test, stage_index, n_sub_next, save_dir, test_n_2d=100):
    problem_dim = len(problem.domain[0])
    print(f"  -> Stage {stage_index}: Generating PoU partition plots for next stage (n_sub={n_sub_next})...")
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
    print(f"     ... plots saved to {filepath}")

def plot_loss_history(loss_hist, l1_steps, l1_hist, stage_index, n_sub, save_dir):
    print(f"  -> Stage {stage_index}: Generating loss history plot (n_sub={n_sub})...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(loss_hist, label='Training Loss', color='tab:blue', alpha=0.8); ax.set_yscale('log'); ax.set_xlabel('Steps'); ax.set_ylabel('Loss', color='tab:blue'); ax.grid(True, which="both", ls="--", alpha=0.5)
    ax2 = ax.twinx(); ax2.plot(l1_steps, l1_hist, '--', label='L1 Error', color='tab:red'); ax2.set_ylabel('L1 Error', color='tab:red')
    fig.suptitle(f'Stage {stage_index}: History (n_sub={n_sub})', fontsize=16); fig.legend(loc="upper right"); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filepath = os.path.join(save_dir, f"loss_history_stage_{stage_index}_nsub_{n_sub}.png"); plt.savefig(filepath, dpi=300); plt.close(fig)
    print(f"     ... plot saved to {filepath}")

if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    problem = LocalHighFreqPoisson1D()
    domain = problem.domain
    problem_dim = len(domain[0])
    print(f"成功推断问题维度: {problem_dim}D")

    FBPINN_STEPS = 30000; FBPINN_LR = 1e-3; POU_EPOCHS = 15000
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S'); save_dir = f"results_{problem.__class__.__name__}_{timestamp}"; os.makedirs(save_dir, exist_ok=True)
    print(f"结果将保存至: {save_dir}")
    
    if problem_dim == 1:
        test_n = 2000; colloc_n = 3000
        x_test = jnp.linspace(domain[0][0], domain[1][0], test_n)[:, None]
        colloc_train = generate_collocation(domain, colloc_n)
    else: # 2D
        test_n_2d = 100; colloc_n = 60
        x_test = generate_collocation(domain, test_n_2d**2)
        colloc_train = generate_collocation(domain, colloc_n**2, method="sobol")
        
    results = {}
    
    BASE_MLP_LARGE = dict(width_size=64, depth=3, activation=jax.nn.tanh)
    BASE_MLP_SMALL = dict(width_size=64, depth=3, activation=jax.nn.tanh)

    # 初始模型训练 (n_sub=1) - Always use the large config
    print("\n" + "="*80); print("===== 初始模型训练: 训练基础PINN (n_sub=1) ====="); print("="*80)
    key, pinn_key = jax.random.split(key)
    
    mlp_conf = dict(in_size=problem_dim, out_size=1, **BASE_MLP_LARGE)
    print(f"    -> Using standard MLP Config (depth={mlp_conf['depth']}, width={mlp_conf['width_size']})")
    
    current_model = FBPINN_PoU(key=pinn_key, domain=domain, num_subdomains=1, mlp_config=mlp_conf, ansatz=problem.ansatz, residual_fn=problem.residual)
    current_model, history = train_fbpinn_simple(key, current_model, problem, colloc_train, FBPINN_LR, FBPINN_STEPS, x_test, problem.exact)
    
    loss_hist, l1_steps, l1_hist = history
    current_l1 = l1_hist[-1]
    current_n_sub = 1
    results[current_n_sub] = {'l1_error': float(current_l1), 'mlp_conf': mlp_conf}

    print(f"\n基础PINN (n_sub=1) 训练完成. L1 Error: {current_l1:.4e}")
    plot_results(current_model, problem, x_test, current_n_sub, current_n_sub, save_dir, test_n_2d=locals().get('test_n_2d', 100))
    plot_loss_history(loss_hist, l1_steps, l1_hist, current_n_sub, current_n_sub, save_dir)
    
    final_l1_steps, final_l1_hist = l1_steps, l1_hist

    N_max = 32
    final_stage_n_sub = 2**int(np.floor(np.log2(N_max - 1))) * 2 if N_max > 2 else 2
    print(f"\n[Experiment Info] Final stage will use a small MLP at n_sub = {final_stage_n_sub}")


    while current_n_sub < N_max:
        next_n_sub = current_n_sub*2
        stage_index = int(np.log2(next_n_sub))

        # 1. 训练PoU网络
        print("\n" + "#"*80)
        print(f"##### STAGE {stage_index}: 基于 n_sub={current_n_sub} 的解, 训练 n_sub={next_n_sub} 的PoU网络 #####")
        print("#"*80)
        y_train_pou = jax.vmap(current_model)(colloc_train).squeeze()
        print(f"\n-- 正在为 {next_n_sub} 个分区训练PoU模型 --")
        key, pou_key = jax.random.split(key)

        if problem_dim == 1:
            pou_net = RBFPOUNet(input_dim=problem_dim, num_centers=next_n_sub, domain=problem.domain, key=pou_key)
            final_pou_params = run_lsgd(pou_net, pou_net.init_params(), colloc_train, y_train_pou, LSGDConfig(n_epochs=POU_EPOCHS))
            learned_window_fn = WindowModule(pou_net=pou_net, params=final_pou_params)
        else:
            print("2D PoU 学习尚未实现，跳过...")
            break
        
        # <<< MODIFIED >>> Collapse check is now disabled.
        # passes_check = check_territory(learned_window_fn, x_test)
        print("    -> [Info] Temporarily skipping territory collapse check.")
        
        # Plot the partitions regardless of whether they pass the check, for diagnosis.
        plot_pou_results(learned_window_fn, problem, x_test, stage_index, next_n_sub, save_dir, test_n_2d=locals().get('test_n_2d', 100))

        # if not passes_check:
        #     print(f"在 {next_n_sub} 个分区处检测到领地丢失，停止增加分区复杂度。")
        #     break
        # else:
        #     print(f"{next_n_sub} 个分区方案健康，是合格的候选分区。")

        # 2. 训练新的FBPINN模型
        print("\n" + "="*80)
        print(f"===== STAGE {stage_index}: FBPINN 训练 (n_sub={next_n_sub}) =====")
        print("="*80)
        key, fbpinn_key = jax.random.split(key)
        
        if next_n_sub == final_stage_n_sub:
            print(f"    -> FINAL STAGE DETECTED (n_sub={next_n_sub}): Using SMALL MLP config.")
            current_base_conf = BASE_MLP_SMALL
        else:
            print(f"    -> Intermediate stage (n_sub={next_n_sub}): Using LARGE MLP config.")
            current_base_conf = BASE_MLP_LARGE
        
        mlp_conf = dict(in_size=problem_dim, out_size=1, **current_base_conf)
        
        next_model = FBPINN_PoU(key=fbpinn_key, domain=domain, num_subdomains=next_n_sub, mlp_config=mlp_conf, ansatz=problem.ansatz, residual_fn=problem.residual, window_fn=learned_window_fn)
        next_model, history = train_fbpinn_simple(key, next_model, problem, colloc_train, FBPINN_LR, FBPINN_STEPS, x_test, problem.exact)

        # Update results and prepare for next iteration
        loss_hist, l1_steps, l1_hist = history
        final_l1_steps, final_l1_hist = l1_steps, l1_hist
        current_l1 = l1_hist[-1]
        results[next_n_sub] = {'l1_error': float(current_l1), 'mlp_conf': mlp_conf}
        
        print(f"\nFBPINN (n_sub={next_n_sub}) 训练完成. L1 Error: {current_l1:.4e}")
        plot_results(next_model, problem, x_test, stage_index, next_n_sub, save_dir)
        plot_loss_history(loss_hist, l1_steps, l1_hist, stage_index, next_n_sub, save_dir)

        current_model = next_model
        current_n_sub = next_n_sub

    # 5. 保存和总结
    stats_path = os.path.join(save_dir, "stats.npz")
    np.savez(stats_path, test_steps=final_l1_steps, test_l1=final_l1_hist)
    print("\n" + "-"*80)
    print(f"最终L1测试历史已保存至: '{stats_path}'")
    print(f"例如: python compare_l1.py {save_dir}")
    print("-" * 80)
    
    print("\n\n" + "#"*80)
    print("##### 算法执行完毕 #####")
    print("#"*80)
    print("\n各层级模型 L1 误差与网络结构总结:")
    sorted_results = sorted(results.items())
    for n_sub_val, res in sorted_results:
        model_type = "基础PINN" if n_sub_val == 1 else "FBPINN"
        conf = res['mlp_conf']
        print(f"模型 ({model_type}, n_sub={n_sub_val:<2}) | "
              f"L1 Error = {res['l1_error']:.4e} | "
              f"MLP Config (depth={conf['depth']}, width={conf['width_size']})")