# -*- coding: utf-8 -*-
"""
FBPINN baseline on a hexagonal domain with ADF boundary enforcement.
- Reference solution from FDM (5-point Laplacian) on a uniform grid.
- Test points == Grid points (exactly the same set of nodes).
- Fixed cosine^2 PoU windows (FBPINN-style).
- Metrics: RMSE and Relative L2, evaluated on the SAME FDM grid nodes (inside hex).
"""

import os
from datetime import datetime
from typing import Sequence, Dict, Tuple, Any, Callable, Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import equinox as eqx
import optax
from tqdm import trange

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- SciPy for sparse linear system in FDM ---
import scipy.sparse as sp
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# =============================================================================
# Geometry and Problem Setup
# =============================================================================
HEXAGON_RADIUS = 1.0
PHASE_SHIFT = np.pi / 2.0

HEXAGON_VERTICES = jnp.array([
    [HEXAGON_RADIUS * jnp.cos(theta + PHASE_SHIFT),
     HEXAGON_RADIUS * jnp.sin(theta + PHASE_SHIFT)]
    for theta in jnp.linspace(0, 2 * np.pi, 7)[:-1]
])

HEXAGON_SEGMENTS = [
    {"p1": HEXAGON_VERTICES[i],
     "p2": HEXAGON_VERTICES[(i + 1) % len(HEXAGON_VERTICES)]}
    for i in range(len(HEXAGON_VERTICES))
]


def is_inside_hexagon(xy: jnp.ndarray) -> jnp.ndarray:
    """Checks if points are inside the convex hexagon via half-planes."""
    points = jnp.atleast_2d(xy)
    x, y = points[:, 0], points[:, 1]
    is_inside = jnp.ones(x.shape[0], dtype=jnp.bool_)
    for i in range(len(HEXAGON_VERTICES)):
        p1 = HEXAGON_VERTICES[i]
        p2 = HEXAGON_VERTICES[(i + 1) % len(HEXAGON_VERTICES)]
        cross_product = (p2[0] - p1[0]) * (y - p1[1]) - (p2[1] - p1[1]) * (x - p1[0])
        is_inside = jnp.logical_and(is_inside, cross_product >= -1e-12)
    return is_inside


# =============================================================================
# ADF (R-functions) for Dirichlet-by-Construction Ansatz
# =============================================================================
def _get_phi_for_segment_jax(point, p1, p2):
    x, y, x1, y1, x2, y2 = point[0], point[1], p1[0], p1[1], p2[0], p2[1]
    dx_seg, dy_seg = x2 - x1, y2 - y1
    L2 = dx_seg**2 + dy_seg**2
    L = jnp.sqrt(L2)
    L = jnp.where(L < 1e-12, 1.0, L)

    f_val = ((x - x1) * dy_seg - (y - y1) * dx_seg) / L
    xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
    t_val = (1 / L) * ((L / 2) ** 2 - ((x - xc) ** 2 + (y - yc) ** 2))
    varphi = jnp.sqrt(t_val**2 + f_val**4)
    phi = jnp.sqrt(f_val**2 + ((varphi - t_val) / 2) ** 2)
    euclidean_dist = jnp.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    return jnp.where(L2 < 1e-12, euclidean_dist, phi)


def _r_function_intersection_jax(phi_list, m_parameter):
    phi_array = jnp.array(phi_list)
    is_on_boundary = jnp.any(phi_array < 1e-9)
    safe_phis = jnp.maximum(phi_array, 1e-12)
    sum_inv_phi_m = jnp.sum(safe_phis ** (-m_parameter))
    combined_phi = jnp.maximum(sum_inv_phi_m, 1e-12) ** (-1.0 / m_parameter)
    return jnp.where(is_on_boundary, 0.0, combined_phi)


def adf_hexagon(point):
    phi_values = [_get_phi_for_segment_jax(point, seg['p1'], seg['p2'])
                  for seg in HEXAGON_SEGMENTS]
    return _r_function_intersection_jax(phi_values, 1.0)


# =============================================================================
# Poisson Problem and PINN Residual
# =============================================================================
class PoissonHexagonComplicatedRHS:
    """-Δu = f  on hexagon, u|∂Ω = 0  (Dirichlet by ADF)."""
    def __init__(self, adf_fn, k_min: float = 2.0, k_max: float = 10.0, omega: float = jnp.pi):
        self.domain = [[-HEXAGON_RADIUS, -HEXAGON_RADIUS],
                       [HEXAGON_RADIUS, HEXAGON_RADIUS]]
        self.adf_fn = adf_fn
        self.k_min, self.k_max, self.omega = float(k_min), float(k_max), omega

    def solution_ansatz(self, model: callable, x: jnp.ndarray) -> jnp.ndarray:
        return (self.adf_fn(x) * model(x)).squeeze()

    def rhs_f(self, x_in: jnp.ndarray) -> jnp.ndarray:
        x, y = x_in[0], x_in[1]
        p = 2.0
        kx = self.k_min + (self.k_max - self.k_min) * ((x + 1) / 2) ** p
        ky = self.k_min + (self.k_max - self.k_min) * ((y + 1) / 2) ** p
        return 100.0 * (jnp.sin(kx * self.omega * x) + jnp.sin(ky * self.omega * y))

    def pointwise_residual(self, model: callable, x: jnp.ndarray) -> jnp.ndarray:
        hess = jax.hessian(self.solution_ansatz, argnums=1)(model, x)
        return -jnp.trace(hess) - self.rhs_f(x)

    def residual(self, model: callable, xy: jnp.ndarray) -> jnp.ndarray:
        if xy.shape[0] == 0:
            return 0.0
        residuals = jax.vmap(self.pointwise_residual, in_axes=(None, 0))(model, xy)
        return jnp.mean(residuals ** 2)


# =============================================================================
# FDM Reference Solver (Uniform Grid + Hex Mask + Dirichlet 0)
# =============================================================================
def solve_poisson_fdm_hexagon(grid_res: int = 201,
                              k_min: float = 2.0,
                              k_max: float = 10.0,
                              omega: float = np.pi) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                             np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a uniform grid on [-1,1]^2. Use a hexagon "inside" mask.
    Assemble 5-point FD for -Δu=f with u=0 outside (Dirichlet).
    Return:
        U (GxG): reference solution on the grid (NaN outside Ω)
        X, Y (GxG): grid coordinates
        eval_points (Neval,2): all grid nodes inside Ω (test points)
        u_eval (Neval,): reference values on those eval points
        inside_mask (GxG bool): inside-hex mask for plotting
    """
    xs = np.linspace(-1.0, 1.0, grid_res)
    ys = np.linspace(-1.0, 1.0, grid_res)
    X, Y = np.meshgrid(xs, ys)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1)
    inside = np.array(is_inside_hexagon(jnp.array(grid_points))).reshape(grid_res, grid_res)

    # mapping from (i,j) to 1D interior index
    idx_map = -np.ones((grid_res, grid_res), dtype=int)
    interior_coords = []
    cnt = 0
    for i in range(grid_res):
        for j in range(grid_res):
            if inside[i, j]:
                idx_map[i, j] = cnt
                interior_coords.append((i, j))
                cnt += 1

    Ne = cnt
    if Ne == 0:
        raise RuntimeError("No interior nodes found for the chosen grid_res.")

    h = 2.0 / (grid_res - 1)  # uniform spacing
    A = lil_matrix((Ne, Ne), dtype=float)
    b = np.zeros((Ne,), dtype=float)

    # RHS function (same as in problem)
    def rhs(x, y):
        p = 2.0
        kx = k_min + (k_max - k_min) * ((x + 1) * 0.5) ** p
        ky = k_min + (k_max - k_min) * ((y + 1) * 0.5) ** p
        return 100.0 * (np.sin(kx * omega * x) + np.sin(ky * omega * y))

    # assemble -Δu = f  ->  (4u - sum(neigh)) / h^2 = f  =>  4u - sum(neigh) = h^2 f
    for row, (i, j) in enumerate(interior_coords):
        A[row, row] = 4.0
        # neighbor offsets: up/down/left/right
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < grid_res and 0 <= nj < grid_res and inside[ni, nj]:
                col = idx_map[ni, nj]
                A[row, col] = -1.0
            else:
                # Dirichlet u=0 outside -> contributes 0 to RHS (no extra term)
                pass
        b[row] = (h ** 2) * rhs(X[i, j], Y[i, j])

    A = csr_matrix(A)
    u_vec = spsolve(A, b)  # reference interior solution

    # Pack back to grid (NaN outside)
    U = np.full((grid_res, grid_res), np.nan, dtype=float)
    for k, (i, j) in enumerate(interior_coords):
        U[i, j] = u_vec[k]

    # Eval points/labels = the SAME grid nodes inside Ω
    eval_points = np.stack([X[inside], Y[inside]], axis=-1)  # (Ne,2)
    u_eval = U[inside]  # (Ne,)

    return U, X, Y, eval_points, u_eval, inside


# =============================================================================
# FBPINN-style Subdomain and Window Function
# =============================================================================
def generate_subdomains(domain, overlap, n_sub_per_dim):
    lowers, uppers = map(jnp.asarray, domain)
    dim = lowers.size
    n_sub_per_dim = [n_sub_per_dim] * dim if isinstance(n_sub_per_dim, int) else n_sub_per_dim

    axes = []
    for i in range(dim):
        if n_sub_per_dim[i] == 1:
            axes.append(jnp.array([(lowers[i] + uppers[i]) / 2.0]))
        else:
            axes.append(jnp.linspace(lowers[i], uppers[i], int(n_sub_per_dim[i])))

    mesh = jnp.meshgrid(*axes, indexing="ij")
    centers = jnp.stack([m.ravel() for m in mesh], axis=-1)

    n_sub_arr = jnp.array(n_sub_per_dim)
    safe_denom = jnp.where(n_sub_arr > 1, n_sub_arr - 1, 1)
    step = (uppers - lowers) / safe_denom

    half_width = (step + overlap) / 2.0
    domain_half_width = (uppers - lowers + overlap) / 2.0
    half_width = jnp.where(n_sub_arr == 1, domain_half_width, half_width)

    half_width_b = jnp.broadcast_to(half_width, centers.shape)
    subdomains = [(c - hw, c + hw) for c, hw in zip(centers, half_width_b)]
    return subdomains


def cosine_window(xmins_all, xmaxs_all, x):
    x = jnp.atleast_2d(x)
    xmins_all = jnp.atleast_2d(xmins_all)
    xmaxs_all = jnp.atleast_2d(xmaxs_all)

    x_ = x[:, None, :]
    xmin = xmins_all[None, :, :]
    xmax = xmaxs_all[None, :, :]

    mu = (xmin + xmax) / 2.0
    sd = (xmax - xmin) / 2.0

    r = (x_ - mu) / sd
    r_clipped = jnp.clip(r, -1.0, 1.0)
    core = ((1.0 + jnp.cos(jnp.pi * r_clipped)) / 2.0) ** 2

    w_raw = jnp.prod(core, axis=-1)
    denom = jnp.maximum(w_raw.sum(axis=1, keepdims=True), 1e-12)
    return w_raw / denom


def create_fbpinn_window_fn(domain, nx, ny, overlap):
    subdomains = generate_subdomains(domain, overlap=overlap, n_sub_per_dim=[nx, ny])
    xmins_all = jnp.array([sub[0] for sub in subdomains])
    xmaxs_all = jnp.array([sub[1] for sub in subdomains])

    @jax.jit
    def window_fn(xy: jnp.ndarray) -> jnp.ndarray:
        return cosine_window(xmins_all, xmaxs_all, xy)

    return window_fn


# =============================================================================
# FBPINN Model and Trainer
# =============================================================================
class FBPINN_PoU(eqx.Module):
    subnets: list
    window_fn: Optional[Callable] = eqx.static_field()

    def __init__(self, key, num_subdomains, mlp_config, window_fn=None):
        self.subnets = [eqx.nn.MLP(**mlp_config, key=k)
                        for k in jax.random.split(key, num_subdomains)]
        self.window_fn = window_fn

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        points = jnp.atleast_2d(x)
        if len(self.subnets) == 1:
            return jax.vmap(self.subnets[0])(points).squeeze()
        partitions = self.window_fn(points)  # (N, n_sub)
        y_subs = jnp.stack([jax.vmap(net)(points) for net in self.subnets], axis=-1).squeeze()
        return jnp.sum(partitions * y_subs, axis=1)


def train_fbpinn(key, model, problem, colloc, lr, steps):
    params, static = eqx.partition(model, eqx.is_array)
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @eqx.filter_jit
    def loss_fn(p, xy):
        return problem.residual(eqx.combine(p, static), xy)

    @eqx.filter_jit
    def step_fn(p, o, all_colloc):
        loss, g = jax.value_and_grad(loss_fn)(p, all_colloc)
        updates, o = opt.update(g, o, p)
        p = eqx.apply_updates(p, updates)
        return p, o, loss

    # warmup JIT
    step_fn(params, opt_state, colloc)

    loss_hist = []
    n_sub = len(model.subnets)
    bar = trange(steps, desc=f"FBPINN (N={n_sub})", dynamic_ncols=True)
    for s in bar:
        params, opt_state, loss = step_fn(params, opt_state, colloc)
        lv = float(loss)
        loss_hist.append(lv)
        if np.isnan(lv) or np.isinf(lv):
            print(f"\nError: Loss became {lv} at step {s}. Aborting.")
            break
        bar.set_postfix(loss=f"{lv:.3e}")

    final_model = eqx.combine(params, static)
    return final_model, jnp.array(loss_hist)


# =============================================================================
# Metrics
# =============================================================================
def compute_rmse_and_rel_l2(u_pred_flat: jnp.ndarray,
                            u_ref_flat: jnp.ndarray) -> Tuple[float, float]:
    err = u_pred_flat - u_ref_flat
    mse = jnp.mean(err ** 2)
    rmse = jnp.sqrt(mse)
    num = jnp.sqrt(jnp.sum(err ** 2))
    denom = jnp.sqrt(jnp.sum(u_ref_flat ** 2))
    rel_l2 = num / jnp.maximum(denom, 1e-12)
    return float(rmse), float(rel_l2)


# =============================================================================
# Plotting and Evaluation
# =============================================================================
def get_grid_dims(n_sub: int) -> Tuple[int, int]:
    if n_sub in [2, 3, 5, 7, 11, 13]:
        return 1, n_sub
    if n_sub == 1:
        return 1, 1
    best_factor = 1
    for i in range(2, int(jnp.sqrt(n_sub)) + 1):
        if n_sub % i == 0:
            best_factor = i
    return (n_sub // best_factor, best_factor)


def plot_results(problem, model, U_fdm, X_fdm, Y_fdm,
                 rmse: float, rel_l2: float,
                 loss_hist, stage_id, n_sub, save_dir):
    print(f"  -> Stage {stage_id} (N={n_sub}): Generating plots...")
    G = U_fdm.shape[0]

    # Predict on full grid for visualization (mask with NaN outside)
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    grid_points = jnp.array(np.stack([X_fdm.ravel(), Y_fdm.ravel()], axis=-1))
    u_pred_flat = jax.vmap(problem.solution_ansatz, in_axes=(None, 0))(model, grid_points)
    u_pred_grid = np.array(u_pred_flat).reshape(G, G)
    mask = ~np.isfinite(U_fdm)  # outside Ω are NaN
    u_pred_grid[mask] = np.nan

    im1 = ax1.imshow(u_pred_grid, extent=[-1, 1, -1, 1],
                     origin='lower', cmap='viridis')
    ax1.set_title(f'Predicted Solution (N={n_sub})')
    fig1.colorbar(im1, ax=ax1)
    ax1.set_aspect('equal', 'box')
    fig1.savefig(os.path.join(save_dir, f"solution_stage_{stage_id}_nsub_{n_sub}.png"), dpi=300)
    plt.close(fig1)

    # Error map
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    error_grid = np.abs(u_pred_grid - np.array(U_fdm))
    im2 = ax2.imshow(error_grid, extent=[-1, 1, -1, 1],
                     origin='lower', cmap='plasma')
    fig2.colorbar(im2, ax=ax2)
    ax2.set_aspect('equal', 'box')
    fig2.savefig(os.path.join(save_dir, f"error_map_stage_{stage_id}_nsub_{n_sub}.png"), dpi=300)
    plt.close(fig2)

    # Loss history
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(loss_hist)
    ax3.set_yscale('log')
    ax3.set_title(f'Loss History (N={n_sub})')
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Log Loss (PDE Residual)')
    ax3.grid(True, which="both", ls="--")
    fig3.savefig(os.path.join(save_dir, f"loss_history_stage_{stage_id}_nsub_{n_sub}.png"), dpi=300)
    plt.close(fig3)


def plot_pou_on_grid(window_fn, stage_id, n_sub, save_dir, grid_res=101):
    nx_plot, ny_plot = get_grid_dims(n_sub)
    print(f"   -> Stage {stage_id} (N={n_sub}): Plotting {nx_plot}x{ny_plot} PoU...")

    x_coords = np.linspace(-1, 1, grid_res)
    y_coords = np.linspace(-1, 1, grid_res)
    X, Y = np.meshgrid(x_coords, y_coords)
    grid_points = jnp.array(np.stack([X.ravel(), Y.ravel()], axis=-1))

    weights_flat = window_fn(grid_points)
    mask = np.array(is_inside_hexagon(grid_points))

    fig, axes = plt.subplots(ny_plot, nx_plot,
                             figsize=(4 * nx_plot, 3.5 * ny_plot), squeeze=False)
    axes = axes.ravel()

    for i in range(n_sub):
        window_grid = np.full((grid_res * grid_res), np.nan)
        window_grid[mask] = np.array(weights_flat[:, i])[mask]
        im = axes[i].imshow(window_grid.reshape(grid_res, grid_res),
                            extent=[-1, 1, -1, 1], origin='lower',
                            cmap='inferno', vmin=0, vmax=1)
        axes[i].set_title(f'Window {i + 1}')
        fig.colorbar(im, ax=axes[i])
        axes[i].set_aspect('equal', 'box')

    fig.suptitle(f'Stage {stage_id}: PoU for {n_sub} Subdomains', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filepath = os.path.join(save_dir, f"pou_stage_{stage_id}_nsub_{n_sub}.png")
    plt.savefig(filepath, dpi=300)
    plt.close(fig)


def save_history(save_dir, stage_id, n_sub, loss_hist, rmse, rel_l2):
    filepath = os.path.join(save_dir, f"history_stage_{stage_id}_nsub_{n_sub}.npz")
    np.savez_compressed(filepath,
                        loss_hist=np.asarray(loss_hist),
                        rmse=np.asarray(rmse),
                        rel_l2=np.asarray(rel_l2))


# =============================================================================
# Main Experiment Runner
# =============================================================================
def run_fbpinn_baseline_experiments(key, problem_class, config):
    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    problem = problem_class(adf_fn=adf_hexagon)

    print("\n" + "=" * 80 + "\n===== FDM Reference Solution (Uniform Grid) =====\n" + "=" * 80)
    (U_fdm, X_fdm, Y_fdm,
     fdm_eval_points, u_fdm_eval, inside_mask) = solve_poisson_fdm_hexagon(
        grid_res=config["grid_res"],
        k_min=config["k_min"], k_max=config["k_max"], omega=np.pi
    )
    print(f"[FDM] grid_res={config['grid_res']}, interior nodes={fdm_eval_points.shape[0]}")

    # >>>> 关键：collocation points 与 test points 一致 <<<<
    use_same_points = True
    if use_same_points:
        colloc_train = jnp.array(fdm_eval_points)  # 完全一致
        print(f"[Collocation] Using SAME {colloc_train.shape[0]} interior grid nodes as training points.")
    else:
        # 备用：若要单独生成，可用网格法或随机法
        raise NotImplementedError

    metrics_history: Dict[int, Tuple[float, float]] = {}

    decomposition_schedule = config["decomposition_schedule"]
    for i, (nx, ny) in enumerate(decomposition_schedule):
        n_sub = nx * ny
        stage_id = i

        print("\n" + "#" * 80 +
              f"\n##### Stage {stage_id}: Training FBPINN with fixed {nx}x{ny} grid (n_sub={n_sub}) #####\n" +
              "#" * 80)

        key, stage_key = jax.random.split(key)
        if n_sub == 1:
            window_fn = None
        else:
            window_fn = create_fbpinn_window_fn(
                domain=problem.domain, nx=nx, ny=ny, overlap=config["pou_overlap"]
            )
            plot_pou_on_grid(window_fn, stage_id, n_sub, save_dir)

        model = FBPINN_PoU(
            key=stage_key, num_subdomains=n_sub,
            mlp_config=config["mlp_conf"], window_fn=window_fn
        )

        trained_model, loss_hist = train_fbpinn(
            stage_key, model, problem, colloc_train,
            config["FBPINN_LR"], config["FBPINN_STEPS"]
        )

        # --- 评估：在同一批 FDM 网格节点（inside）上 ---
        print(f"\n--- Evaluating Stage {stage_id} (n_sub={n_sub}) on FDM grid nodes ---")
        u_pinn_on_grid = jax.vmap(problem.solution_ansatz, in_axes=(None, 0))(
            trained_model, jnp.array(fdm_eval_points)
        )
        rmse, rel_l2 = compute_rmse_and_rel_l2(u_pinn_on_grid, jnp.array(u_fdm_eval))
        print(f" -> RMSE: {rmse:.4e} | Relative L2: {rel_l2:.4e}")
        metrics_history[n_sub] = (float(rmse), float(rel_l2))

        plot_results(problem, trained_model, U_fdm, X_fdm, Y_fdm,
                     rmse, rel_l2, loss_hist, stage_id, n_sub, save_dir)
        save_history(save_dir, stage_id, n_sub, loss_hist, rmse, rel_l2)

    # --- Summary Plot ---
    print("\n" + "=" * 80 + "\n===== RMSE & Relative L2 Summary =====\n" + "=" * 80)
    sorted_keys = sorted(metrics_history.keys())
    if len(sorted_keys) > 0:
        n_subs_list = sorted_keys
        rmse_list = [metrics_history[k][0] for k in sorted_keys]
        rel_l2_list = [metrics_history[k][1] for k in sorted_keys]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(n_subs_list, rmse_list, 'o-', label='RMSE')
        ax.plot(n_subs_list, rel_l2_list, 's--', label='Relative L2')
        ax.set_xlabel('Number of Subdomains (N)')
        ax.set_ylabel('Metric value')
        ax.set_title('RMSE & Relative L2 vs. Number of Subdomains (FBPINN Baseline, FDM ref)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(n_subs_list)
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.grid(True, which="both", ls="--")
        ax.legend()
        plt.tight_layout()
        filepath = os.path.join(save_dir, "metrics_vs_n_sub_baseline.png")
        plt.savefig(filepath, dpi=300)
        plt.close(fig)
        print(f"Metrics vs. N_sub summary saved to {filepath}")


# =============================================================================
# Entry
# =============================================================================
if __name__ == '__main__':
    config = {
        # FDM grid
        "grid_res": 201,     # 测试点 & 网格点分辨率（可改）
        "k_min": 2.0,
        "k_max": 10.0,

        # Training
        "FBPINN_STEPS": 30000,
        "FBPINN_LR": 1e-3,

        # Model
        "mlp_conf": dict(in_size=2, out_size=1, width_size=16, depth=2, activation=jnp.tanh),

        # FBPINN Decomposition
        "decomposition_schedule": [
            (3, 3),  # 9 subdomains
        ],
        "pou_overlap": 0.3,
    }

    problem_class = PoissonHexagonComplicatedRHS
    print(f"\nSolving problem: {problem_class.__name__}")
    print("Reference = FDM on uniform grid; Train/Eval points = SAME grid nodes inside hex.\n")

    key = jax.random.PRNGKey(42)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config["save_dir"] = f"results_{problem_class.__name__}_FBPINN_BASELINE_FDM_{timestamp}"
    os.makedirs(config["save_dir"], exist_ok=True)
    print(f"Results will be saved to: {config['save_dir']}\n")

    run_fbpinn_baseline_experiments(key, problem_class, config)

    print("\n\n" + "#" * 80 + "\n##### Execution finished. #####\n" + "#" * 80)
    print(f"All results and plots have been saved to: '{config['save_dir']}'")
