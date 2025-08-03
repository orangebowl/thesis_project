import jax, jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from functools import partial
import os, sys
import equinox as eqx
import optax
from typing import Sequence, Dict, Tuple, Any, Callable, Optional
from tqdm import trange
import numpy as np

# FEniCS imports for reference solution
from dolfin import *
import mshr

from jax import config
config.update("jax_enable_x64", True)

# --- Geometry and Problem Setup ---
HEXAGON_RADIUS = 1.0
MESH_RESOLUTION = 64
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

def solve_poisson_fem_hexagon_fenics(resolution: int = 64, grid_res: int = 201, k_min: float = 2.0, k_max: float = 10.0, omega: float = np.pi, p: float = 2.0):
    """Calculates the FEM reference solution using FEniCS."""
    R = 1.0
    phase = np.pi/2
    verts = [Point(R*np.cos(k*2*np.pi/6 + phase), R*np.sin(k*2*np.pi/6 + phase)) for k in range(6)]
    domain = mshr.Polygon(verts)
    mesh = mshr.generate_mesh(domain, resolution)
    print(f"[FEM] Mesh: {mesh.num_vertices()} vertices, {mesh.num_cells()} cells")
    V = FunctionSpace(mesh, "Lagrange", 1)
    u, v = TrialFunction(V), TestFunction(V)
    bc = DirichletBC(V, Constant(0.0), "on_boundary")
    class RHS(UserExpression):
        def eval(self, values, x):
            kx = k_min + (k_max-k_min)*((x[0]+1)/2)**p
            ky = k_min + (k_max-k_min)*((x[1]+1)/2)**p
            values[0] = 100.0*(np.sin(kx*omega*x[0]) + np.sin(ky*omega*x[1]))
        def value_shape(self): return ()
    f = RHS(degree=5)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx
    u_sol = Function(V)
    solve(a == L, u_sol, bc, solver_parameters={"linear_solver": "cg", "preconditioner": "hypre_amg"})
    print("[FEM] Poisson equation solved.")
    xs = np.linspace(-1, 1, grid_res)
    ys = np.linspace(-1, 1, grid_res)
    X, Y = np.meshgrid(xs, ys)
    U = np.full_like(X, np.nan)
    for i in range(grid_res):
        for j in range(grid_res):
            p_check = Point(float(X[i, j]), float(Y[i, j]))
            if domain.inside(p_check):
                U[i, j] = u_sol(p_check)
    return U, X, Y

def is_inside_hexagon(xy: jnp.ndarray) -> jnp.ndarray:
    """Checks if points are inside the hexagon (JAX version) - Fixed."""
    points = jnp.atleast_2d(xy)
    x, y = points[:, 0], points[:, 1]
    
    is_inside = jnp.ones(x.shape[0], dtype=jnp.bool_)
    for i in range(len(HEXAGON_VERTICES)):
        p1 = HEXAGON_VERTICES[i]
        p2 = HEXAGON_VERTICES[(i + 1) % len(HEXAGON_VERTICES)]
        cross_product = (p2[0] - p1[0]) * (y - p1[1]) - (p2[1] - p1[1]) * (x - p1[0])
        is_inside &= (cross_product >= -1e-9)
    return is_inside

# --- ADF and Problem Definition ---
def _get_phi_for_segment_jax(point, p1, p2):
    x, y, x1, y1, x2, y2 = point[0], point[1], p1[0], p1[1], p2[0], p2[1]
    dx_seg, dy_seg = x2 - x1, y2 - y1
    L_squared = dx_seg**2 + dy_seg**2; L = jnp.sqrt(L_squared); L = jnp.where(L < 1e-9, 1.0, L)
    f_val = ((x - x1) * dy_seg - (y - y1) * dx_seg) / L; xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
    t_val = (1 / L) * ((L / 2)**2 - ((x - xc)**2 + (y - yc)**2)); varphi_sq = t_val**2 + f_val**4
    varphi = jnp.sqrt(varphi_sq); phi_term2_num = varphi - t_val; phi_term2_sq = (phi_term2_num / 2)**2
    phi_sq_arg = f_val**2 + phi_term2_sq; phi = jnp.sqrt(phi_sq_arg)
    euclidean_dist = jnp.sqrt((x - x1)**2 + (y - y1)**2)
    return jnp.where(L_squared < 1e-12, euclidean_dist, phi)

def _r_function_intersection_jax(phi_list, m_parameter):
    phi_array = jnp.array(phi_list); is_on_boundary = jnp.any(phi_array < 1e-9)
    safe_phis = jnp.maximum(phi_array, 1e-12); sum_inv_phi_m = jnp.sum(safe_phis**(-m_parameter))
    safe_sum = jnp.maximum(sum_inv_phi_m, 1e-12); combined_phi = safe_sum**(-1.0 / m_parameter)
    return jnp.where(is_on_boundary, 0.0, combined_phi)

def adf_hexagon(point):
    """Approximate Distance Function (ADF) for the hexagon."""
    phi_values = [_get_phi_for_segment_jax(point, seg['p1'], seg['p2']) for seg in HEXAGON_SEGMENTS]
    return _r_function_intersection_jax(phi_values, 1.0)

class PoissonHexagonComplicatedRHS:
    """Defines the Poisson problem with a complicated source term."""
    def __init__(self, adf_fn, k_min: float = 2.0, k_max: float = 10.0, omega: float = jnp.pi):
        self.domain = [[-HEXAGON_RADIUS, -HEXAGON_RADIUS], [HEXAGON_RADIUS, HEXAGON_RADIUS]]
        self.adf_fn = adf_fn
        self.k_min, self.k_max, self.omega = float(k_min), float(k_max), omega

    def solution_ansatz(self, model: callable, x: jnp.ndarray) -> jnp.ndarray:
        return (self.adf_fn(x) * model(x)).squeeze()

    def rhs_f(self, x_in: jnp.ndarray) -> jnp.ndarray:
        x, y = x_in[0], x_in[1]
        p = 2.0
        kx = self.k_min + (self.k_max - self.k_min) * ((x + 1) / 2) ** p
        ky = self.k_min + (self.k_max - self.k_min) * ((y + 1) / 2) ** p
        return 100 * (jnp.sin(kx * self.omega * x) + jnp.sin(ky * self.omega * y))

    def pointwise_residual(self, model: callable, x: jnp.ndarray) -> jnp.ndarray:
        hess = jax.hessian(self.solution_ansatz, argnums=1)(model, x)
        return -jnp.trace(hess) - self.rhs_f(x)

    def residual(self, model: callable, xy: jnp.ndarray) -> jnp.ndarray:
        if xy.shape[0] == 0:
            return 0.0
        residuals = jax.vmap(self.pointwise_residual, in_axes=(None, 0))(model, xy)
        return jnp.mean(residuals ** 2)

def generate_collocation_points(n_candidates: int, is_inside_fn: Callable, domain_bounds: list):
    """Generates collocation points within the domain and applies a mask."""
    n_side = int(np.ceil(np.sqrt(n_candidates)))
    xs = np.linspace(domain_bounds[0][0], domain_bounds[1][0], n_side)
    ys = np.linspace(domain_bounds[0][1], domain_bounds[1][1], n_side)
    X, Y = np.meshgrid(xs, ys)
    grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
    
    inside_mask = is_inside_fn(grid_points)
    inside_pts = grid_points[inside_mask]
    
    print(f"[Collocation Points] Total {grid_points.shape[0]} points -> Kept {inside_pts.shape[0]} interior points.")
    if inside_pts.shape[0] == 0:
        raise ValueError("Failed to generate any collocation points inside the domain. Check 'is_inside_hexagon' logic.")
    return inside_pts

# --- FBPINN-style Subdomain and Window Function (FIXED for N=1 case and broadcasting) ---
def generate_subdomains(domain, overlap, n_sub_per_dim):
    """Generates overlapping subdomains based on FBPINN paper, with fixes for N=1 and broadcasting."""
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
    """FBPINN-style Cosine^2 window function."""
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
    core = ((1.0 + jnp.cos(jnp.pi * r_clipped)) / 2.0)**2
    
    w_raw = jnp.prod(core, axis=-1)
    denom = jnp.maximum(w_raw.sum(axis=1, keepdims=True), 1e-12)
    return w_raw / denom

def create_fbpinn_window_fn(domain, nx, ny, overlap):
    """Creates a fixed window function for FBPINN."""
    subdomains = generate_subdomains(domain, overlap=overlap, n_sub_per_dim=[nx, ny])
    xmins_all = jnp.array([sub[0] for sub in subdomains])
    xmaxs_all = jnp.array([sub[1] for sub in subdomains])

    @jax.jit
    def window_fn(xy: jnp.ndarray) -> jnp.ndarray:
        return cosine_window(xmins_all, xmaxs_all, xy)
        
    return window_fn

# --- FBPINN Model and Trainer ---
class FBPINN_PoU(eqx.Module):
    subnets: list
    window_fn: Optional[Callable] = eqx.static_field()

    def __init__(self, key, num_subdomains, mlp_config, window_fn=None):
        self.subnets = [eqx.nn.MLP(**mlp_config, key=k) for k in jax.random.split(key, num_subdomains)]
        self.window_fn = window_fn

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        points = jnp.atleast_2d(x)
        if len(self.subnets) == 1:
            return jax.vmap(self.subnets[0])(points).squeeze()

        partitions = self.window_fn(points)
        y_subs = jnp.stack([jax.vmap(net)(points) for net in self.subnets], axis=-1).squeeze()
        return jnp.sum(partitions * y_subs, axis=1)

def train_fbpinn(key, model, problem, colloc, lr, steps):
    params, static = eqx.partition(model, eqx.is_array)
    opt = optax.adam(lr)
    opt_state = opt.init(params)
    
    @eqx.filter_jit
    def loss_fn(p, xy): return problem.residual(eqx.combine(p, static), xy)
    
    @eqx.filter_jit
    def step_fn(p, o, all_colloc):
        loss, g = jax.value_and_grad(loss_fn)(p, all_colloc)
        updates, o = opt.update(g, o, p)
        p = eqx.apply_updates(p, updates)
        return p, o, loss
        
    print("JIT compiling FBPINN trainer...", end="", flush=True); step_fn(params, opt_state, colloc); print(" Done.")
    
    loss_hist = []
    n_sub = len(model.subnets)
    bar = trange(steps, desc=f"FBPINN (N={n_sub})", dynamic_ncols=True)
    
    for s in bar:
        params, opt_state, loss = step_fn(params, opt_state, colloc)
        loss_val = float(loss)
        loss_hist.append(loss_val)
        if np.isnan(loss_val) or np.isinf(loss_val):
            print(f"\nError: Loss became {loss_val} at step {s}. Aborting."); break
        bar.set_postfix(loss=f"{loss_val:.3e}")
        
    final_model = eqx.combine(params, static)
    return final_model, jnp.array(loss_hist)

# --- Plotting and Evaluation (FIXED read-only error and titles) ---
def get_grid_dims(n_sub: int) -> Tuple[int, int]:
    if n_sub in [2, 3, 5, 7, 11, 13]: return 1, n_sub
    if n_sub == 1: return 1, 1
    best_factor = 1
    for i in range(2, int(jnp.sqrt(n_sub)) + 1):
        if n_sub % i == 0: best_factor = i
    return (n_sub // best_factor, best_factor)

def plot_results(problem, model, U_fem, X_fem, Y_fem, l1_error, loss_hist, stage_id, n_sub, save_dir):
    print(f"  -> Stage {stage_id} (N={n_sub}): Generating and saving plots...")
    grid_res = U_fem.shape[0]
    
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    grid_points = jnp.array(np.stack([X_fem.ravel(), Y_fem.ravel()], axis=-1))
    u_pred_flat = jax.vmap(problem.solution_ansatz, in_axes=(None, 0))(model, grid_points)
    
    u_pred_grid = np.array(u_pred_flat).reshape(grid_res, grid_res)
    mask = ~np.isfinite(U_fem)
    u_pred_grid[mask] = np.nan
    
    im1 = ax1.imshow(u_pred_grid, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
    ax1.set_title(f'Predicted Solution (N={n_sub})')
    fig1.colorbar(im1, ax=ax1)
    ax1.set_aspect('equal', 'box')
    fig1.savefig(os.path.join(save_dir, f"solution_stage_{stage_id}_nsub_{n_sub}.png"), dpi=300)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    error_grid = np.abs(u_pred_grid - np.array(U_fem))
    im2 = ax2.imshow(error_grid, extent=[-1, 1, -1, 1], origin='lower', cmap='plasma')
    ax2.set_title(f'Absolute Error vs. FEM (L1 = {l1_error:.3e})')
    fig2.colorbar(im2, ax=ax2)
    ax2.set_aspect('equal', 'box')
    fig2.savefig(os.path.join(save_dir, f"error_map_stage_{stage_id}_nsub_{n_sub}.png"), dpi=300)
    plt.close(fig2)
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(loss_hist)
    ax3.set_yscale('log')
    ax3.set_title(f'Loss History (N={n_sub})')
    ax3.set_xlabel('Training Steps'); ax3.set_ylabel('Log Loss (PDE Residual)')
    ax3.grid(True, which="both", ls="--")
    fig3.savefig(os.path.join(save_dir, f"loss_history_stage_{stage_id}_nsub_{n_sub}.png"), dpi=300)
    plt.close(fig3)

def plot_pou_on_grid(window_fn, stage_id, n_sub, save_dir, grid_res=101):
    nx_plot, ny_plot = get_grid_dims(n_sub)
    print(f"  -> Stage {stage_id} (N={n_sub}): Plotting {nx_plot}x{ny_plot} PoU...")
    x_coords = np.linspace(-1, 1, grid_res); y_coords = np.linspace(-1, 1, grid_res)
    X, Y = np.meshgrid(x_coords, y_coords)
    grid_points = jnp.array(np.stack([X.ravel(), Y.ravel()], axis=-1))
    
    weights_flat = window_fn(grid_points)
    mask = is_inside_hexagon(grid_points)
    
    fig, axes = plt.subplots(ny_plot, nx_plot, figsize=(4 * nx_plot, 3.5 * ny_plot), squeeze=False)
    axes = axes.ravel()
    for i in range(n_sub):
        window_grid = jnp.full((grid_res * grid_res), jnp.nan)
        window_grid = window_grid.at[mask].set(weights_flat[mask, i]).reshape(grid_res, grid_res)
        im = axes[i].imshow(np.asarray(window_grid).T, extent=[-1, 1, -1, 1], origin='lower', cmap='inferno', vmin=0, vmax=1)
        axes[i].set_title(f'Window {i+1}')
        fig.colorbar(im, ax=axes[i])
        axes[i].set_aspect('equal', 'box')
    
    fig.suptitle(f'Stage {stage_id}: Fixed FBPINN PoU ({n_sub} Subdomains)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, f"pou_stage_{stage_id}_nsub_{n_sub}.png"), dpi=300)
    plt.close(fig)

def save_history(save_dir, stage_id, n_sub, loss_hist, l1_error, rel_l1_error):
    filepath = os.path.join(save_dir, f"history_stage_{stage_id}_nsub_{n_sub}.npz")
    np.savez_compressed(filepath, loss_hist=np.asarray(loss_hist), l1_error=np.asarray(l1_error), rel_l1_error=np.asarray(rel_l1_error))

# --- Main Experiment Runner ---
def run_fbpinn_baseline_experiments(key, problem_class, config):
    save_dir = config["save_dir"]
    problem = problem_class(adf_fn=adf_hexagon)

    print("\n" + "="*80 + "\n===== Calculating FEM Reference Solution =====\n" + "="*80)
    U_fem, X_fem, Y_fem = solve_poisson_fem_hexagon_fenics()
    
    interior_mask = np.isfinite(U_fem)
    fem_interior_points = jnp.array(np.stack([X_fem[interior_mask], Y_fem[interior_mask]], -1))
    u_fem_flat = jnp.array(U_fem[interior_mask])

    print("\n" + "="*80 + "\n===== Generating Collocation Points =====\n" + "="*80)
    key, train_key = jax.random.split(key)
    colloc_train = generate_collocation_points(
        config["colloc_n_points"], is_inside_hexagon, problem.domain
    )

    l1_errors_history = {}
    
    decomposition_schedule = config["decomposition_schedule"]

    for i, (nx, ny) in enumerate(decomposition_schedule):
        n_sub = nx * ny
        stage_id = i 

        print("\n" + "#"*80 +
              f"\n##### Stage {stage_id}: Training FBPINN with fixed {nx}x{ny} grid (n_sub={n_sub}) #####\n" +
              "#"*80)

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
        if len(loss_hist) < config["FBPINN_STEPS"] and np.any(np.isnan(loss_hist)):
             print(f"\nTraining for n_sub={n_sub} stopped early.")
        else:
            print(f"\nTraining for n_sub={n_sub} complete. Final Loss: {loss_hist[-1]:.4e}")

        print(f"\n--- Evaluating Stage {stage_id} (n_sub={n_sub}) Error ---")
        u_pinn_flat = jax.vmap(problem.solution_ansatz, in_axes=(None, 0))(
            trained_model, fem_interior_points
        )
        l1_error = jnp.mean(jnp.abs(u_pinn_flat - u_fem_flat))
        rel_l1_error = l1_error / jnp.mean(jnp.abs(u_fem_flat))
        print(f" -> L1 Error: {l1_error:.4e} | Relative L1 Error: {rel_l1_error:.4e}")
        l1_errors_history[n_sub] = (float(l1_error), float(rel_l1_error))

        plot_results(problem, trained_model, U_fem, X_fem, Y_fem, l1_error, loss_hist, stage_id, n_sub, save_dir)
        save_history(save_dir, stage_id, n_sub, loss_hist, l1_error, rel_l1_error)

    # --- Final Summary Plot ---
    print("\n" + "="*80 + "\n===== L1 Error Summary =====\n" + "="*80)
    sorted_keys = sorted(l1_errors_history.keys())
    
    if len(sorted_keys) > 1:
        n_subs_list = sorted_keys
        errors_list = [l1_errors_history[k][0] for k in sorted_keys]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(n_subs_list, errors_list, 'o-', label='L1 Error vs. FEM')
        ax.set_xlabel('Number of Subdomains (N)')
        ax.set_ylabel('L1 Error')
        ax.set_title('L1 Error vs. Number of Subdomains (FBPINN Baseline)')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xticks(n_subs_list); ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.grid(True, which="both", ls="--")
        ax.legend()
        plt.tight_layout()
        filepath = os.path.join(save_dir, "l1_error_vs_n_sub_baseline.png")
        plt.savefig(filepath, dpi=300)
        plt.close(fig)
        print(f"L1 error vs. N_sub summary plot saved to {filepath}")
    else:
        print("Only one data point, skipping summary plot generation.")


if __name__ == '__main__':
    config = {
        # Training parameters
        "FBPINN_STEPS": 20000,
        "FBPINN_LR": 1e-3,
        
        # Data and Model parameters (Aligned with adaptive version)
        "colloc_n_points": 10000,
        "mlp_conf": dict(in_size=2, out_size=1, width_size=16, depth=2, activation=jnp.tanh),
        
        # FBPINN Decomposition Config
        "decomposition_schedule": [
            (3, 3),  # 9 subdomains
        ],
        "pou_overlap": 0.3, # Overlap between subdomains
    }

    problem_class = PoissonHexagonComplicatedRHS
    print(f"\nSolving problem: {problem_class.__name__}")
    print("Using fixed FBPINN (Cosine) domain decomposition (Baseline)")

    key = jax.random.PRNGKey(42)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config["save_dir"] = f"results_{problem_class.__name__}_FBPINN_BASELINE_{timestamp}"
    os.makedirs(config["save_dir"], exist_ok=True)
    print(f"Results will be saved to: {config['save_dir']}\n")

    run_fbpinn_baseline_experiments(key, problem_class, config)

    print("\n\n" + "#"*80 + "\n##### Execution finished. #####\n" + "#"*80)
    print(f"All results and plots have been saved to: '{config['save_dir']}'")
