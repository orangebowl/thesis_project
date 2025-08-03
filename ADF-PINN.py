import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from functools import partial
from tqdm import trange, tqdm
import os
from datetime import datetime

# ==============================================================================
# SECTION 1: 几何定义 (ADF) 和 问题定义 (Poisson)
# 这部分和之前一样，用于描述L形域和其上的PDE
# ==============================================================================

# --- 1.1 ADF (Approximate Distance Function) for L-Shape ---
_L_SHAPE_VERTICES = jnp.array([
    [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0],
    [1.0, 0.0],  [0.0, 0.0],   [0.0, 1.0]
])
_L_SHAPE_SEGMENTS = []
for i in range(len(_L_SHAPE_VERTICES)):
    p1 = _L_SHAPE_VERTICES[i]
    p2 = _L_SHAPE_VERTICES[(i + 1) % len(_L_SHAPE_VERTICES)]
    _L_SHAPE_SEGMENTS.append({'p1': p1, 'p2': p2})
_R_FUNC_M = 6.0

def _get_phi_for_segment_jax(point, p1, p2):
    x, y = point[0], point[1]
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
    dx_seg, dy_seg = x2 - x1, y2 - y1
    L_squared = dx_seg**2 + dy_seg**2
    L = jnp.sqrt(L_squared)
    L = jnp.where(L < 1e-9, 1.0, L)
    f_val = ((x - x1) * dy_seg - (y - y1) * dx_seg) / L
    xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
    t_val = (1 / L) * ((L / 2)**2 - ((x - xc)**2 + (y - yc)**2))
    varphi = jnp.sqrt(t_val**2 + f_val**4)
    phi = jnp.sqrt(f_val**2 + ((varphi - t_val) / 2)**2)
    euclidean_dist = jnp.sqrt((x - x1)**2 + (y - y1)**2)
    return jnp.where(L_squared < 1e-12, euclidean_dist, phi)

def _r_function_intersection_jax(phi_list, m_parameter):
    phi_array = jnp.array(phi_list)
    is_on_boundary = jnp.any(phi_array < 1e-9)
    safe_phis = jnp.maximum(phi_array, 1e-12)
    sum_inv_phi_m = jnp.sum(safe_phis**(-m_parameter))
    safe_sum = jnp.maximum(sum_inv_phi_m, 1e-12)
    combined_phi = safe_sum**(-1.0 / m_parameter)
    return jnp.where(is_on_boundary, 0.0, combined_phi)

def adf_l_shape(point):
    phi_values = [_get_phi_for_segment_jax(point, seg['p1'], seg['p2']) for seg in _L_SHAPE_SEGMENTS]
    return _r_function_intersection_jax(phi_values, _R_FUNC_M)

# --- 1.2 Problem Definition for Poisson Equation on L-Shaped Domain ---
class PoissonLShaped:
    def __init__(self):
        self.domain = [[-1., -1.], [1., 1.]]
        self.dim = 2

    @staticmethod
    def ansatz(model: callable, x: jnp.ndarray) -> jnp.ndarray:
        D_x = adf_l_shape(x)
        N_x = model(x)
        return (D_x * N_x).squeeze()

    @staticmethod
    def exact(x: jnp.ndarray) -> jnp.ndarray:
        return adf_l_shape(x) * jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])

    def rhs_f(self, x: jnp.ndarray) -> jnp.ndarray:
        hessian_u_exact = jax.hessian(self.exact)(x)
        return -jnp.trace(hessian_u_exact)
    
    def pointwise_residual(self, model: callable, x: jnp.ndarray) -> jnp.ndarray:
        def full_solution(point):
            return self.ansatz(model, point)
        hessian_u_model = jax.hessian(full_solution)(x)
        laplacian_u_model = jnp.trace(hessian_u_model)
        pde_lhs = -laplacian_u_model
        pde_rhs = self.rhs_f(x)
        return pde_lhs - pde_rhs

    def loss_fn(self, model: callable, xy: jnp.ndarray) -> jnp.ndarray:
        res_all = jax.vmap(self.pointwise_residual, in_axes=(None, 0))(model, xy)
        return jnp.mean(res_all**2)

# ==============================================================================
# SECTION 2: 数据采样和绘图工具
# ==============================================================================

def is_inside_L_shape(xy: jnp.ndarray) -> jnp.ndarray:
    x, y = xy[:, 0], xy[:, 1]
    in_bounding_box = (x >= -1) & (x <= 1) & (y >= -1) & (y <= 1)
    in_excluded_quadrant = (x > 0) & (y > 0)
    return in_bounding_box & ~in_excluded_quadrant

def generate_collocation_L_shape(n_points: int, key: jax.random.PRNGKey) -> jnp.ndarray:
    print(f"Generating {n_points} collocation points inside L-shaped domain...")
    bounding_box_min = jnp.array([-1.0, -1.0])
    bounding_box_max = jnp.array([1.0, 1.0])
    collected_points, num_collected = [], 0
    pbar = tqdm(total=n_points, desc="  Rejection Sampling")
    while num_collected < n_points:
        num_candidates = int((n_points - num_collected) * 1.5) + 100
        key, subkey = jax.random.split(key)
        candidates = jax.random.uniform(subkey, shape=(num_candidates, 2), minval=bounding_box_min, maxval=bounding_box_max)
        mask = is_inside_L_shape(candidates)
        inside_points = candidates[mask]
        num_found = inside_points.shape[0]
        if num_found > 0:
            actual_needed = min(num_found, needed := n_points - num_collected)
            collected_points.append(inside_points[:actual_needed])
            num_collected += actual_needed
            pbar.update(actual_needed)
    pbar.close()
    return jnp.concatenate(collected_points, axis=0)

def plot_results(model, problem, save_dir):
    print("Generating final plots...")
    # Generate a dense grid of test points for high-quality visualization
    key = jax.random.PRNGKey(99)
    x_test = generate_collocation_L_shape(20000, key)
    u_exact = jax.vmap(problem.exact)(x_test)
    u_pred = jax.vmap(lambda m, p: m.ansatz(m, p), in_axes=(None, 0))(model, x_test)
    error = jnp.abs(u_pred - u_exact)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    vmax, vmin = jnp.max(u_exact), jnp.min(u_exact)
    
    im1 = axes[0].scatter(x_test[:, 0], x_test[:, 1], c=u_pred, cmap='viridis', s=2, vmin=vmin, vmax=vmax)
    axes[0].set_title('PINN Predicted Solution'); fig.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].scatter(x_test[:, 0], x_test[:, 1], c=u_exact, cmap='viridis', s=2, vmin=vmin, vmax=vmax)
    axes[1].set_title('Exact Solution'); fig.colorbar(im2, ax=axes[1])

    im3 = axes[2].scatter(x_test[:, 0], x_test[:, 1], c=error, cmap='Reds', s=2)
    axes[2].set_title('Absolute Error'); fig.colorbar(im3, ax=axes[2])
    
    for ax in axes:
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(problem.domain[0][0], problem.domain[1][0])
        ax.set_ylim(problem.domain[0][1], problem.domain[1][1])
    
    fig.suptitle('PINN Solution for Poisson Equation on L-Shaped Domain', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filepath = os.path.join(save_dir, "pinn_solution_l_shape.png")
    plt.savefig(filepath, dpi=300); plt.close(fig)
    print(f"Result plots saved to {filepath}")

if __name__ == '__main__':
    # --- 1. 配置参数 ---
    config = {
        "learning_rate": 1e-3,
        "steps": 50000,
        "colloc_points": 10000,
        "mlp_width": 128,
        "mlp_depth": 4,
        "seed": 42,
        "eval_every": 2000,
    }

    # --- 2. 初始化 ---
    key = jax.random.PRNGKey(config["seed"])
    model_key, data_key, test_key = jax.random.split(key, 3)
    
    problem = PoissonLShaped()
    
    model = eqx.nn.MLP(
        in_size=problem.dim,
        out_size=1,
        width_size=config["mlp_width"],
        depth=config["mlp_depth"],
        activation=jnp.tanh,
        key=model_key
    )
    
    collocation_points = generate_collocation_L_shape(config["colloc_points"], data_key)

    # --- 3. 定义训练步骤 (Corrected Pattern) ---

    # [CORRECTION 1] Partition the model into parameters (arrays) and static structure.
    params, static = eqx.partition(model, eqx.is_array)
    
    optimizer = optax.adam(config["learning_rate"])
    # Initialize the optimizer with the parameters only.
    opt_state = optimizer.init(params)

    @eqx.filter_jit
    def make_step(current_params, current_opt_state, points):
        # Define the loss function with respect to the params, which jax.grad can handle.
        def loss_for_grad(p):
            # Recombine the model inside the JIT-compiled function.
            recombined_model = eqx.combine(p, static)
            return problem.loss_fn(recombined_model, points)
        
        loss, grads = jax.value_and_grad(loss_for_grad)(current_params)
        
        updates, new_opt_state = optimizer.update(grads, current_opt_state)
        new_params = eqx.apply_updates(current_params, updates)
        
        return new_params, new_opt_state, loss

    # --- 4. 执行训练循环 (Corrected Pattern) ---
    print("\nStarting PINN training...")
    bar = trange(config["steps"], desc="Training PINN", leave=True)
    for step in bar:
        # [CORRECTION 2] The loop now updates `params`, not the whole `model`.
        params, opt_state, loss_val = make_step(params, opt_state, collocation_points)
        
        if step % config["eval_every"] == 0 or step == config["steps"] - 1:
            bar.set_postfix(loss=f"{loss_val:.4e}")

    print("Training finished.")

    # [CORRECTION 3] Recombine the final trained params with the static structure.
    final_model = eqx.combine(params, static)

    # --- 5. 评估和可视化 ---
    print("Evaluating final model...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_directory = f"results_PINN_L_Shape_{timestamp}"
    os.makedirs(save_directory, exist_ok=True)
    print(f"Results will be saved to: {save_directory}")
    
    eval_points = generate_collocation_L_shape(20000, test_key)
    u_exact_eval = jax.vmap(problem.exact)(eval_points)
    
    # Pass the final model to the wrapper for evaluation
    def predict_wrapper(m, p):
        return problem.ansatz(m, p) # The ansatz needs the callable model part
    u_pred_eval = jax.vmap(predict_wrapper, in_axes=(None, 0))(final_model, eval_points)
    
    l2_error = jnp.linalg.norm(u_pred_eval - u_exact_eval) / jnp.linalg.norm(u_exact_eval)
    print(f"\nFinal Relative L2 Error: {l2_error:.4e}")

    # For plotting, create a final model object that knows its ansatz
    class FinalModel(eqx.Module):
        net: eqx.Module
        ansatz_fn: Callable
        
        def __init__(self, final_net, problem_ansatz):
            self.net = final_net
            self.ansatz_fn = problem_ansatz
            
        def __call__(self, x):
            return self.ansatz_fn(self.net, x)

    final_model_for_plot = FinalModel(final_model, problem.ansatz)
    
    def plot_results_final(model_to_plot, problem, save_dir):
        # (This plotting function from the previous step is correct and does not need changes)
        print("Generating final plots...")
        key = jax.random.PRNGKey(99)
        x_test = generate_collocation_L_shape(20000, key)
        u_exact = jax.vmap(problem.exact)(x_test)
        u_pred = jax.vmap(model_to_plot)(x_test)
        error = jnp.abs(u_pred - u_exact)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        vmax, vmin = jnp.max(u_exact), jnp.min(u_exact)
        
        im1 = axes[0].scatter(x_test[:, 0], x_test[:, 1], c=u_pred, cmap='viridis', s=2, vmin=vmin, vmax=vmax, rasterized=True)
        axes[0].set_title('PINN Predicted Solution'); fig.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].scatter(x_test[:, 0], x_test[:, 1], c=u_exact, cmap='viridis', s=2, vmin=vmin, vmax=vmax, rasterized=True)
        axes[1].set_title('Exact Solution'); fig.colorbar(im2, ax=axes[1])

        im3 = axes[2].scatter(x_test[:, 0], x_test[:, 1], c=error, cmap='Reds', s=2, rasterized=True)
        axes[2].set_title('Absolute Error'); fig.colorbar(im3, ax=axes[2])
        
        for ax in axes:
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(problem.domain[0][0], problem.domain[1][0])
            ax.set_ylim(problem.domain[0][1], problem.domain[1][1])
        
        fig.suptitle('PINN Solution for Poisson Equation on L-Shaped Domain', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        filepath = os.path.join(save_dir, "pinn_solution_l_shape.png")
        plt.savefig(filepath, dpi=300); plt.close(fig)
        print(f"Result plots saved to {filepath}")

    plot_results_final(final_model_for_plot, problem, save_directory)