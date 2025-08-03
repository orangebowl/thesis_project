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
from tqdm import trange, tqdm
import numpy as np
import dataclasses
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import griddata

from jax import config
config.update("jax_enable_x64", True)

project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import matplotlib.pyplot as plt
from dolfin import *
import mshr

def solve_poisson_fem_lshape_fenics(resolution: int = 64,
                                    grid_res: int = 201,
                                    k_min: float = 2.0,
                                    k_max: float = 10.0,
                                    omega: float = np.pi,
                                    p: float = 2.0):
    if mshr is None:
        raise RuntimeError("FEniCS / mshr not found. Install them before calling this routine.")

    R_full = mshr.Rectangle(Point(-1, -1), Point(1, 1))
    R_cut  = mshr.Rectangle(Point(0, 0), Point(1, 1))
    domain = R_full - R_cut                     # CSG difference → L‑shape

    mesh = mshr.generate_mesh(domain, resolution)
    print(f"[FEM] mesh: {mesh.num_vertices()} verts, {mesh.num_cells()} cells")

    V  = FunctionSpace(mesh, "Lagrange", 1)
    u  = TrialFunction(V)
    v  = TestFunction(V)
    bc = DirichletBC(V, Constant(0.0), "on_boundary")

    class RHS(UserExpression):
        def eval(self, values, x):
            kx = k_min + (k_max - k_min) * (((x[0] + 1)/2) ** p)
            ky = k_min + (k_max - k_min) * (((x[1] + 1)/2) ** p)
            values[0] = 100.0 * (np.sin(kx * omega * x[0]) + np.sin(ky * omega * x[1]))
        def value_shape(self):
            return ()
    f = RHS(degree=5)

    a = dot(grad(u), grad(v)) * dx
    L = f * v * dx

    u_sol = Function(V)
    solve(a == L, u_sol, bc,
          solver_parameters={"linear_solver": "cg",
                             "preconditioner": "hypre_amg"})
    print("[FEM] Poisson solve done.")

    # ---- interpolate onto regular grid (with NaNs outside) ----
    xs = np.linspace(-1, 1, grid_res)
    ys = np.linspace(-1, 1, grid_res)
    X, Y = np.meshgrid(xs, ys)
    U = np.full_like(X, np.nan)
    inside_bool = np.logical_and.reduce([X >= -1, X <= 1, Y >= -1, Y <= 1, ~(np.logical_and(X > 0, Y > 0))])
    for i in range(grid_res):
        for j in range(grid_res):
            if inside_bool[i, j]:
                U[i, j] = u_sol(Point(float(X[i, j]), float(Y[i, j])))

    # ---- collect interior node values from the mesh (for L1 error) ----
    coords = mesh.coordinates()
    mask_int = np.array([(-1 <= px <= 1) and (-1 <= py <= 1) and not (px > 0 and py > 0) for px, py in coords])
    pts_int  = coords[mask_int]
    u_flat   = u_sol.vector().get_local()[mask_int]

    return U, X, Y, pts_int, u_flat
_R_FUNC_M = 1.0

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

L_SHAPE_BBOX = (-1.0, 1.0)  # convenience alias

def is_inside_lshape(xy: jax.Array) -> jax.Array:
    """Mask of points inside the L‑shape (vectorised, works on (N,2) or (2,) arrays)."""
    pts = jnp.atleast_2d(xy)
    x, y = pts[:, 0], pts[:, 1]
    in_square   = (x >= -1.0) & (x <= 1.0) & (y >= -1.0) & (y <= 1.0)
    not_cut_out = ~((x > 0.0) & (y > 0.0))               # remove upper‑right corner
    return in_square & not_cut_out


def adf_lshape(point: jax.Array) -> jax.Array:
    """Simple, positive, *approximate* distance‑to‑boundary for the L‑shape.

    It takes the *minimum* distance to all axis‑aligned edges that bound the
    domain.  This is cheap and perfectly adequate for the multiplicative
    ansatz ϕ(x)·NN(x) we use in the PINN.
    """
    x, y = point[0], point[1]

    # outer square boundaries
    d_left, d_right   = x + 1.0, 1.0 - x
    d_bottom, d_top  = y + 1.0, 1.0 - y

    # inner boundaries introduced by the cut‑out
    d_inner_v = jnp.where(y >= 0.0,  jnp.abs(x), jnp.inf)  # vertical x = 0
    d_inner_h = jnp.where(x >= 0.0,  jnp.abs(y), jnp.inf)  # horizontal y = 0

    return jnp.minimum(jnp.minimum(jnp.minimum(d_left, d_right),
                                   jnp.minimum(d_bottom, d_top)),
                       jnp.minimum(d_inner_v, d_inner_h))

def check_territory(window_fn: Callable[[jax.Array], jax.Array],
                           x_test: jax.Array,
                           dominance_threshold: float = 0.8) -> bool:
    """Exactly the logic you had before, but:
       • we *only* care about test points inside the physical domain; and
       • a partition that collapses **outside** the L‑shape is *not* an error.
    """
    weights = window_fn(x_test)                # (N_pts, N_sub)
    num_parts = weights.shape[1]

    inside_mask = is_inside_lshape(x_test)     # (N_pts,) bool
    dominant_idx = jnp.argmax(weights, axis=1)

    def _has_territory_inside(i):
        return jnp.any((dominant_idx == i) & inside_mask)

    has_inside = jax.vmap(_has_territory_inside)(jnp.arange(num_parts))
    missing = jnp.where(~has_inside)[0]        # partitions with no interior turf
    if missing.size > 0:
        print(f"       - HEALTH CHECK FAILED: partitions {missing} own no points *inside* the L‑shape.")
        return False

    # Strength check (only over interior points)
    for i in range(num_parts):
        mask_i = (dominant_idx == i) & inside_mask
        if not jnp.any(mask_i):
            continue  # lives outside ⇒ ignore
        if jnp.max(weights[mask_i, i]) < dominance_threshold:
            print(f"       - HEALTH CHECK FAILED: partition {i} is weak inside L‑shape (peak {jnp.max(weights[mask_i, i]):.3e} < {dominance_threshold}).")
            return False

    print(f"       - Health check passed for {num_parts} partitions (interior‑aware).")
    return True


class PoissonLshapeComplicatedRHS:
    """Same physics as your *Hexagon* version, but for the L‑shape domain."""

    def __init__(self, adf_fn: Callable[[jax.Array], jax.Array],
                 k_min: float = 2.0,
                 k_max: float = 10.0,
                 omega: float = jnp.pi):
        self.domain = [[-1.0, -1.0], [1.0, 1.0]]
        self.dim = 2
        self.adf_fn = adf_fn
        self.k_min = float(k_min)
        self.k_max = float(k_max)
        self.omega = omega

    # ----- neural‑network solution ansatz ---------------------------------
    def solution_ansatz(self, model: Callable[[jax.Array], jax.Array], x: jax.Array) -> jax.Array:
        return (self.adf_fn(x) * model(x)).squeeze()

    # ----- RHS f(x) -------------------------------------------------------
    def rhs_f(self, x_in: jax.Array) -> jax.Array:
        x, y = x_in[0], x_in[1]
        p = 2.0
        kx = self.k_min + (self.k_max - self.k_min) * ((x + 1) / 2) ** p
        ky = self.k_min + (self.k_max - self.k_min) * ((y + 1) / 2) ** p
        return 100.0 * (jnp.sin(kx * self.omega * x) + jnp.sin(ky * self.omega * y))

    # ----- PDE residual ---------------------------------------------------
    def pointwise_residual(self, model: Callable[[jax.Array], jax.Array], x: jax.Array) -> jax.Array:
        hess = jax.hessian(self.solution_ansatz, argnums=1)(model, x)
        return -jnp.trace(hess) - self.rhs_f(x)

    def residual(self, model: Callable[[jax.Array], jax.Array], xy: jax.Array) -> jax.Array:
        residuals = jax.vmap(self.pointwise_residual, in_axes=(None, 0))(model, xy)
        return jnp.mean(residuals ** 2)

def generate_collocation_points(n_candidates: int,
                                     is_inside_fn: Callable,
                                     domain_bounds: list):
    """
    网格采样版：
      1. 构造 n_side × n_side 的规则网格（n_side² ≥ n_candidates）；
      2. 仅保留位于几何域内的节点；
      3. 返回 (N_in, 2) JAX 数组，其中 N_in ≤ n_candidates。
    """
    # ---------- 1) 在外接正方形内生成规则网格 ----------
    domain_min, domain_max = map(float, domain_bounds[0]), map(float, domain_bounds[1])
    domain_min, domain_max = np.array(list(domain_min)), np.array(list(domain_max))

    # 令 n_side² >= n_candidates
    n_side = int(np.ceil(np.sqrt(n_candidates)))
    xs = np.linspace(domain_min[0], domain_max[0], n_side)
    ys = np.linspace(domain_min[1], domain_max[1], n_side)
    X, Y = np.meshgrid(xs, ys)                        # 形状 (n_side, n_side)
    grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)  # (n_side², 2)

    # ---------- 2) 用 is_inside_fn 过滤域外点 ----------
    inside_mask = is_inside_fn(grid_points)
    inside_pts  = grid_points[inside_mask]

    kept = int(inside_pts.shape[0])
    total = int(grid_points.shape[0])
    print(f"[Grid sampling] total {total} points → kept {kept} inside "
          f"({100*kept/total:.1f}% kept).")

    return inside_pts
class FBPINN_PoU(eqx.Module):
    subnets: list
    window_fn: Optional[Callable]

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

# —— 1. Glorot 初始化 --------------------------------------------------------
def _glorot_uniform(key, shape):
    fan_in, fan_out = shape[0], shape[1]
    lim = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-lim, maxval=+lim)

# —— 2. 一维 MLP 权重初始化 --------------------------------------------------
def _init_mlp_1d(key, hidden: Sequence[int], out_dim: int) -> Dict[str, Any]:
    """
    生成 {'W0', 'b0', 'W1', 'b1', …, 'W{L}', 'b{L}'} 字典，
    其中 L = len(hidden) ；输入维度固定为 1。
    """
    dims  = [1] + list(hidden) + [out_dim]
    n_lay = len(dims) - 1
    keys  = jax.random.split(key, n_lay)

    params = {}
    for i, (m, n) in enumerate(zip(dims[:-1], dims[1:])):
        params[f"W{i}"] = _glorot_uniform(keys[i], (m, n))
        # —— 给 bias 加微小扰动，避免完全对称 —— #
        params[f"b{i}"] = 1e-2 * jax.random.normal(keys[i], (n,))
    return params

# —— 3. 一维 MLP 前向 --------------------------------------------------------
def _mlp_forward_1d(params: Dict[str, Any], x: jnp.ndarray) -> jnp.ndarray:
    """
    x shape: (N, 1)  or  (N,)  → 自动扩为 (N,1)
    """
    if x.ndim == 1:
        x = x[:, None]

    h   = x
    L   = (len(params) // 2) - 1          # 隐藏层数
    for i in range(L):
        h = jnp.tanh(h @ params[f"W{i}"] + params[f"b{i}"])
    return h @ params[f"W{L}"] + params[f"b{L}"]   # (N, out_dim)

# —— 4. 主类：SepMLPPOUNet ---------------------------------------------------
class SepMLPPOUNet:
    """
    构建 2D 分区的可分离 PoU：
        logits(x,y) = (MLP_x(x) + MLP_y(y)) / τ
        soft-max(logits) → 权重 (N_pts, nx*ny)
    """
    def __init__(
        self,
        nx: int, ny: int,
        hidden: Sequence[int] = (32, 32),
        tau: float = 0.1,
        key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    ):
        self.nx, self.ny   = int(nx), int(ny)
        self.num_experts   = self.nx * self.ny
        self.tau           = float(tau)

        kx, ky             = jax.random.split(key)
        self.param_x       = _init_mlp_1d(kx, hidden, self.nx)
        self.param_y       = _init_mlp_1d(ky, hidden, self.ny)

    # ——— 参数打包 / 解包 ——————————————————————————
    def init_params(self) -> Dict[str, Any]:
        return {"x": self.param_x, "y": self.param_y}

    # ——— 前向：输出 (N, nx*ny) 权重 ——————————————————
    def forward(self, params: Dict[str, Any], xy: jnp.ndarray) -> jnp.ndarray:
        xy = jnp.atleast_2d(xy)             # (N,2)
        x, y = xy[:, :1], xy[:, 1:]

        z_x = _mlp_forward_1d(params["x"], x)    # (N, nx)
        z_y = _mlp_forward_1d(params["y"], y)    # (N, ny)

        # 外和 → (N, nx, ny)  再 reshape
        logits = (z_x[:, :, None] + z_y[:, None, :]) / self.tau
        logits = logits.reshape(xy.shape[0], -1)

        # 数值稳定 soft-max
        logits = logits - jnp.max(logits, axis=1, keepdims=True)
        phi    = jax.nn.softmax(logits, axis=1)
        return phi                               # (N, nx*ny)

class WindowModule(eqx.Module):
    pou_net: Any = eqx.static_field(); params: Dict
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray: return self.pou_net.forward(self.params, x)

@dataclasses.dataclass
class LSGDConfig:
    n_epochs: int = 5000; lr: float = 1e-4; lam_init: float = 0.01; rho: float = 0.99; n_stag: int = 300

def _design_matrix(x: jnp.ndarray, degree: int) -> jnp.ndarray:
    if x.ndim == 1:
        x = x.reshape(-1, 2)
    x1 = x[:, 0]
    x2 = x[:, 1]
    features = [jnp.ones_like(x1)]
    if degree > 0:
        for d in range(1, degree + 1):
            for i in range(d + 1):
                j = d - i
                features.append((x1**i) * (x2**j))
    return jnp.stack(features, axis=1)

def fit_local_polynomials(x, y, w, lam):
    poly_degree = 2
    A = _design_matrix(x, poly_degree)
    y_col = y.reshape(-1, 1)
    I = jnp.eye(A.shape[1])
    def _solve(weights, A_matrix, y_vector):
        Aw = A_matrix * weights.reshape(-1, 1)
        M = A_matrix.T @ Aw
        M += lam * I
        b = (Aw.T @ y_vector).squeeze()
        coeffs = jnp.linalg.solve(M, b)
        return coeffs
    return jax.vmap(_solve, in_axes=(1, None, None), out_axes=0)(w, A, y_col)

def _predict_from_coeffs(x, coeffs, partitions, degree=2):
    A = _design_matrix(x, degree)
    y_cent = A @ coeffs.T
    return jnp.sum(partitions * y_cent, 1)

def run_lsgd(pou_net, initial_params, x, y, cfg: LSGDConfig):
    params = initial_params

    # --- 原有的 loss_fn & valgrad_fn ------------
    def loss_fn(p, lam):
        part  = pou_net.forward(p, x)
        coeff = fit_local_polynomials(x, y, part, lam)
        pred  = _predict_from_coeffs(x, coeff, part)
        return jnp.mean((pred - y)**2)
    valgrad_fn = jax.jit(jax.value_and_grad(loss_fn))
    # -------------------------------------------

    opt = optax.adam(cfg.lr)
    opt_state = opt.init(params)
    lam = jnp.array(cfg.lam_init)
    best, stag = jnp.inf, 0

    bar = trange(cfg.n_epochs, desc=f"PoU-LSGD (N={pou_net.num_experts})")

    for ep in bar:
        # 每一步算一次 loss & grad
        loss_val, grads = valgrad_fn(params, lam)

        # 仅在第 0 步打印一次诊断（也可 ep % 100 == 0 定期看）
        if ep == 0:
            # ① 权重分布
            part = pou_net.forward(params, x[:256])
            print("\n[DEBUG] part.std   =", float(part.std()))

            # ② 多项式系数幅值
            coeff = fit_local_polynomials(x[:256], y[:256], part, lam)
            print("[DEBUG] max|coeff| =", float(jnp.max(jnp.abs(coeff))))

            # ③ 梯度范数
            g_flat, _ = jax.flatten_util.ravel_pytree(grads)
            print("[DEBUG] grad L2    =", float(jnp.linalg.norm(g_flat)))
            print("------------------------------------------------------\n")

        # ---- 原有的 Adam 更新 & 早停逻辑 ---------------------------
        updates, opt_state = opt.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)

        bar.set_postfix(loss=f"{loss_val:.3e}")
        if loss_val < best - 1e-12:
            best, stag = loss_val, 0
        else:
            stag += 1
        if stag > cfg.n_stag:
            lam *= cfg.rho
            stag = 0
    return params

def train_pinn_with_batching(key, model, problem, colloc, lr, steps, batch_size):
    params, static = eqx.partition(model, eqx.is_array)
    opt = optax.adam(lr)
    opt_state = opt.init(params)
    @eqx.filter_jit
    def loss_fn(p, xy): return problem.residual(eqx.combine(p, static), xy)
    @eqx.filter_jit
    def step_fn(p, o, batch):
        loss, g = jax.value_and_grad(loss_fn)(p, batch); updates, o = opt.update(g, o, p)
        p = eqx.apply_updates(p, updates); return p, o, loss
    print("JIT compiling PINN trainer...", end="", flush=True); step_fn(params, opt_state, colloc[:batch_size]); print(" Done")
    loss_hist = []
    bar = trange(steps, desc="PINN (N=1, Batched)", dynamic_ncols=True)
    for s in bar:
        key, step_key = jax.random.split(key); batch_indices = jax.random.choice(step_key, len(colloc), (batch_size,), replace=False)
        params, opt_state, loss = step_fn(params, opt_state, colloc[batch_indices]); loss_hist.append(float(loss))
        if jnp.isnan(loss) or jnp.isinf(loss):
            print(f"\nERROR: Loss is NaN/Inf at step {s}. Aborting stage."); break
        bar.set_postfix(loss=f"{loss:.3e}")
    final_model = eqx.combine(params, static); return final_model, jnp.array(loss_hist)

def train_fbpinn_no_batching(key, model, problem, colloc, lr, steps):
    params, static = eqx.partition(model, eqx.is_array)
    opt = optax.adam(lr)
    opt_state = opt.init(params)
    @eqx.filter_jit
    def loss_fn(p, xy): return problem.residual(eqx.combine(p, static), xy)
    @eqx.filter_jit
    def step_fn(p, o, all_colloc):
        loss, g = jax.value_and_grad(loss_fn)(p, all_colloc); updates, o = opt.update(g, o, p)
        p = eqx.apply_updates(p, updates); return p, o, loss
    print("JIT compiling FBPINN trainer...", end="", flush=True); step_fn(params, opt_state, colloc); print(" Done")
    loss_hist = []
    n_sub = len(model.subnets); bar = trange(steps, desc=f"FBPINN (N={n_sub}, Full-batch)", dynamic_ncols=True)
    for s in bar:
        params, opt_state, loss = step_fn(params, opt_state, colloc); loss_hist.append(float(loss))
        if jnp.isnan(loss) or jnp.isinf(loss):
            print(f"\nERROR: Loss is NaN/Inf at step {s}. Aborting."); break
        bar.set_postfix(loss=f"{loss:.3e}")
    final_model = eqx.combine(params, static); return final_model, jnp.array(loss_hist)

def plot_reference_solution(u_fdm_grid, X, Y, save_dir):
    print("  -> Generating FEM reference solution plot...")
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    domain_extent = [np.min(X), np.max(X), np.min(Y), np.max(Y)]
    im = ax.imshow(u_fdm_grid, extent=domain_extent, origin='lower', cmap='viridis')
    ax.set_title('FEM Reference Solution')
    fig.colorbar(im, ax=ax)
    ax.set_aspect('equal', adjustable='box'); ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.suptitle('Finite Difference Method (FEM) Reference Solution', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filepath = os.path.join(save_dir, "reference_fdm_solution.png"); plt.savefig(filepath, dpi=300); plt.close(fig)
    print(f"      ... Reference plot saved to {filepath}")

def plot_error_map(u_pinn_grid, u_fdm_grid, X, Y, l1_error, stage_index, n_sub, save_dir):
    print(f"  -> Stage {stage_index}: Generating error map plot (n_sub={n_sub})...")
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    error_grid = np.abs(u_pinn_grid - u_fdm_grid)
    domain_extent = [-1, 1, -1, 1]
    im = ax.imshow(error_grid , extent=[-1, 1, -1, 1], origin='lower', cmap='plasma')
    ax.set_title(f'Absolute Error vs. FEM (L1 = {l1_error:.3e})')
    fig.colorbar(im, ax=ax)
    ax.set_aspect('equal', adjustable='box'); ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.suptitle(f'Stage {stage_index}: FBPINN Error Map ({n_sub} subdomains)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filepath = os.path.join(save_dir, f"error_map_stage_{stage_index}_nsub_{n_sub}.png"); plt.savefig(filepath, dpi=300); plt.close(fig)
    print(f"      ... Error plot saved to {filepath}")

def get_grid_dims(n_sub: int) -> Tuple[int, int]:
    if n_sub == 1: return 1, 1
    if n_sub in [2, 3, 5, 7, 11, 13]: return 1, n_sub
    best_factor = 1
    for i in range(2, int(jnp.sqrt(n_sub)) + 1):
        if n_sub % i == 0: best_factor = i
    return (best_factor, n_sub // best_factor) if best_factor != 1 else (1, n_sub)

def plot_results_on_grid(problem, model, stage_index, n_sub, save_dir, grid_res=201):
    print(f"  -> Stage {stage_index}: Generating grid plot (n_sub={n_sub})...")
    x_coords = np.linspace(-1, 1, grid_res); y_coords = np.linspace(-1, 1, grid_res)
    X, Y = np.meshgrid(x_coords, y_coords)
    grid_points = jnp.array(np.stack([X.ravel(), Y.ravel()], axis=-1))
    cpu_device = jax.devices('cpu')[0]
    model_cpu = jax.tree_util.tree_map(lambda z: jax.device_put(z, cpu_device) if eqx.is_array(z) else z, model)
    grid_points_cpu = jax.device_put(grid_points, cpu_device)
    u_pred_flat = jax.vmap(problem.solution_ansatz, in_axes=(None, 0))(model_cpu, grid_points_cpu)
    u_pred_grid = u_pred_flat.reshape(grid_res, grid_res)
    mask = is_inside_lshape(grid_points).reshape(grid_res, grid_res)
    u_pred_grid = jnp.where(mask, u_pred_grid, jnp.nan)
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    im = ax.imshow(np.asarray(u_pred_grid), extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
    ax.set_title('Predicted Solution')
    fig.colorbar(im, ax=ax)
    ax.set_aspect('equal', adjustable='box'); ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.suptitle(f'Stage {stage_index}: FBPINN Predicted Solution ({n_sub} subdomains)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filepath = os.path.join(save_dir, f"solution_stage_{stage_index}_nsub_{n_sub}.png"); plt.savefig(filepath, dpi=300); plt.close(fig)
    print(f"      ... Plot saved to {filepath}")

def plot_pou_on_grid(window_fn, stage_index, n_sub_next, save_dir, grid_res=101):
    print(f"  -> Stage {stage_index}: Generating PoU grid plots (for {n_sub_next} subdomains)...")
    x_coords = np.linspace(-1, 1, grid_res); y_coords = np.linspace(-1, 1, grid_res)
    X, Y = np.meshgrid(x_coords, y_coords)
    grid_points = jnp.array(np.stack([X.ravel(), Y.ravel()], axis=-1))
    weights_flat = window_fn(grid_points)
    mask = is_inside_lshape(grid_points)
    num_windows = weights_flat.shape[1]; nx, ny = get_grid_dims(num_windows)
    fig, axes = plt.subplots(ny, nx, figsize=(4 * nx, 3.5 * ny), squeeze=False); axes = axes.ravel()
    for i in range(num_windows):
        window_grid = jnp.full((grid_res, grid_res), jnp.nan)
        window_grid = window_grid.ravel().at[mask].set(weights_flat[:, i][mask]).reshape(grid_res, grid_res)
        im = axes[i].imshow(np.asarray(window_grid).T, extent=[-1, 1, -1, 1], origin='lower', cmap='inferno', vmin=0, vmax=1)
        axes[i].set_title(f'Window {i+1}'); fig.colorbar(im, ax=axes[i])
        axes[i].set_aspect('equal', adjustable='box')
    for j in range(num_windows, len(axes)): axes[j].axis('off')
    fig.suptitle(f'Stage {stage_index}: Learned PoU for {n_sub_next} Subdomains', fontsize=16); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filepath = os.path.join(save_dir, f"pou_stage_{stage_index}_nsub_next_{n_sub_next}.png"); plt.savefig(filepath, dpi=300); plt.close(fig)
    print(f"      ... Plot saved to {filepath}")

def plot_pou_3d(window_fn, stage_index, n_sub_next, save_dir, grid_res=101):
    print(f"  -> Stage {stage_index}: Generating 3D PoU plots (for {n_sub_next} subdomains)...")
    x_coords = np.linspace(-1, 1, grid_res)
    y_coords = np.linspace(-1, 1, grid_res)
    X, Y = np.meshgrid(x_coords, y_coords)
    grid_points = jnp.array(np.stack([X.ravel(), Y.ravel()], axis=-1))
    weights_flat = window_fn(grid_points)
    mask = ~np.asarray(is_inside_lshape(grid_points))
    num_windows = weights_flat.shape[1]
    nx, ny = get_grid_dims(num_windows)
    fig = plt.figure(figsize=(5 * nx, 4.5 * ny))
    fig.suptitle(f'Stage {stage_index}: 3D Learned PoU for {n_sub_next} Subdomains', fontsize=16)
    for i in range(num_windows):
        ax = fig.add_subplot(ny, nx, i + 1, projection='3d')
        window_grid_flat = weights_flat[:, i]
        window_grid_flat = jnp.where(mask, jnp.nan, window_grid_flat)
        Z = np.asarray(window_grid_flat).reshape(grid_res, grid_res)
        ax.plot_surface(X, Y, Z, cmap='inferno', vmin=0, vmax=1, rstride=1, cstride=1, antialiased=True)
        ax.set_title(f'Window {i+1}')
        ax.set_zlim(0, 1)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('Weight')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filepath = os.path.join(save_dir, f"pou_3d_stage_{stage_index}_nsub_next_{n_sub_next}.png")
    plt.savefig(filepath, dpi=300); plt.close(fig)
    print(f"      ... 3D PoU plot saved to {filepath}")

def plot_adf_3d(adf_fn, save_dir, grid_res=101):
    print("  -> Generating 3D ADF plot...")
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    x_coords = np.linspace(-1, 1, grid_res)
    y_coords = np.linspace(-1, 1, grid_res)
    X, Y = np.meshgrid(x_coords, y_coords)
    grid_points = jnp.array(np.stack([X.ravel(), Y.ravel()], axis=-1))
    adf_values_flat = jax.vmap(adf_fn)(grid_points)
    Z = np.asarray(adf_values_flat).reshape(grid_res, grid_res)
    mask = ~np.asarray(is_inside_lshape(grid_points)).reshape(grid_res, grid_res)
    Z = np.where(mask, np.nan, Z)
    ax.plot_surface(X, Y, Z, cmap='viridis', rstride=1, cstride=1, antialiased=True)
    ax.set_title('Approximate Distance Function (ADF)')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('ADF value')
    filepath = os.path.join(save_dir, "adf_3d_plot.png")
    plt.savefig(filepath, dpi=300); plt.close(fig)
    print(f"      ... 3D ADF plot saved to {filepath}")

def plot_pou_sum_verification(window_fn, stage_index, n_sub_next, save_dir, grid_res=101):
    print(f"  -> Stage {stage_index}: Generating PoU Sum Verification Plot (for {n_sub_next} subdomains)...")
    x_coords = np.linspace(-1, 1, grid_res)
    y_coords = np.linspace(-1, 1, grid_res)
    X, Y = np.meshgrid(x_coords, y_coords)
    grid_points = jnp.array(np.stack([X.ravel(), Y.ravel()], axis=-1))
    weights_flat = window_fn(grid_points)
    sum_of_weights = jnp.sum(weights_flat, axis=1)
    mask = ~np.asarray(is_inside_lshape(grid_points))
    sum_grid_flat = jnp.where(mask, jnp.nan, sum_of_weights)
    Z = np.asarray(sum_grid_flat).reshape(grid_res, grid_res)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', vmin=0.99, vmax=1.01)
    ax.set_title(f'Sum of All Window Functions (Stage {stage_index}, N={n_sub_next})')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('Sum of Weights')
    ax.set_zlim(0, 1.5)
    filepath = os.path.join(save_dir, f"pou_sum_verification_stage_{stage_index}_nsub_next_{n_sub_next}.png")
    plt.savefig(filepath, dpi=300); plt.close(fig)
    print(f"      ... Sum verification plot saved to {filepath}")

def plot_adf_window_and_laplacian(window_fn, adf_fn, stage_index, n_sub_next, save_dir, grid_res=71):
    print(f"  -> Stage {stage_index}: Generating ADF*Window and Laplacian plots (N={n_sub_next})...")
    x_coords = np.linspace(-1, 1, grid_res)
    y_coords = np.linspace(-1, 1, grid_res)
    X, Y = np.meshgrid(x_coords, y_coords)
    grid_points = jnp.array(np.stack([X.ravel(), Y.ravel()], axis=-1))
    weights_flat = window_fn(grid_points)
    adf_values_flat = jax.vmap(adf_fn)(grid_points)
    num_windows = weights_flat.shape[1]
    def get_laplacian_fn(i):
        g_i = lambda x: adf_fn(x) * window_fn(jnp.atleast_2d(x))[:, i].squeeze()
        return jax.jit(jax.vmap(lambda p: jnp.trace(jax.hessian(g_i)(p))))
    laplacian_fns = [get_laplacian_fn(i) for i in range(num_windows)]
    for i in range(num_windows):
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle(f'Diagnostics for Window {i+1} (Stage {stage_index}, N={n_sub_next})', fontsize=16)
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        product_values_flat = adf_values_flat * weights_flat[:, i]
        mask = ~np.asarray(is_inside_lshape(grid_points))
        Z1 = np.where(mask, np.nan, np.asarray(product_values_flat)).reshape(grid_res, grid_res)
        ax1.plot_surface(X, Y, Z1, cmap='viridis', rstride=1, cstride=1, antialiased=True)
        ax1.set_title(r'ADF(x) $\times$ Window(x)')
        ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('Value')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        laplacian_values_flat = laplacian_fns[i](grid_points)
        Z2 = np.where(mask, np.nan, np.asarray(laplacian_values_flat)).reshape(grid_res, grid_res)
        norm = mcolors.SymLogNorm(linthresh=0.1, vmin=Z2[~np.isnan(Z2)].min(), vmax=Z2[~np.isnan(Z2)].max())
        ax2.plot_surface(X, Y, Z2, cmap='coolwarm', rstride=1, cstride=1, antialiased=True, norm=norm)
        ax2.set_title(r'$\nabla^2$ (ADF $\times$ Window)')
        ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('Laplacian Value')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filepath = os.path.join(save_dir, f"diag_win_{i+1}_stage_{stage_index}.png")
        plt.savefig(filepath, dpi=300)
        plt.close(fig)
    print(f"      ... Diagnostic plots saved for all {num_windows} windows.")

def plot_loss_history(loss_hist, stage_index, n_sub, save_dir):
    print(f"  -> Stage {stage_index}: Generating loss history plot (n_sub={n_sub})...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(loss_hist, label='Training Loss (PDE Residual)', color='tab:blue')
    ax.set_yscale('log'); ax.set_xlabel('Steps'); ax.set_ylabel('Loss')
    ax.grid(True, which="both", ls="--", alpha=0.5); ax.legend(loc='best')
    fig.suptitle(f'Stage {stage_index}: Loss History ({n_sub} subdomains)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filepath = os.path.join(save_dir, f"loss_history_stage_{stage_index}_nsub_{n_sub}.png"); plt.savefig(filepath, dpi=300); plt.close(fig)

def save_npz_history(save_dir, stage_index, n_sub, loss_hist, l1_error, rel_l1_error):
    filepath = os.path.join(save_dir, f"history_stage_{stage_index}_nsub_{n_sub}.npz")
    np.savez_compressed(filepath, loss_hist=np.asarray(loss_hist), l1_error=np.asarray(l1_error), rel_l1_error=np.asarray(rel_l1_error))

def run_hierarchical_fbpinn(key, problem_class, config):
    save_dir = config["save_dir"]
    adf_fn = adf_lshape

    problem = problem_class(adf_fn=adf_fn)
    MESH_RESOLUTION = 64
    U_fem, X_fem, Y_fem, interior_pts, u_flat = solve_poisson_fem_lshape_fenics(MESH_RESOLUTION)


    # 统一变量名（沿用后续旧代码）
    U_fdm, X_fdm, Y_fdm = U_fem, X_fem, Y_fem
    interior_mask           = np.isfinite(U_fdm)                    # (grid_res,grid_res)
    grid_interior_coords    = np.stack([X_fdm[interior_mask],      # (N,2)
                                        Y_fdm[interior_mask]], -1)
    fdm_interior_points     = grid_interior_coords                  # ← 覆盖旧变量
    u_fdm_flat              = U_fdm[interior_mask]                  # (N,)

    plot_reference_solution(U_fdm, X_fdm, Y_fdm, save_dir)
    plot_adf_3d(adf_fn, save_dir)

    # JAX 版本
    fdm_interior_points_jax = jnp.array(fdm_interior_points)
    u_fdm_flat_jax          = jnp.array(u_fdm_flat)

    # ------------------------------------------------------------------ #
    #  生成测试点 / 训练点                                               #
    # ------------------------------------------------------------------ #
    key, test_key, train_key = jax.random.split(key, 3)
    x_test = generate_collocation_points(
                config["test_n_points"],      # 依旧传同一个数字
                is_inside_lshape,
                problem.domain
            )

    colloc_train = generate_collocation_points(
                    config["colloc_n_points"],
                    is_inside_lshape,
                    problem.domain
                )

    l1_errors_history = {}

    # ======================  Stage‑0: 单域 PINN  ========================= #
    print("\n" + "="*80 + "\n===== Stage 0: Training Base PINN Model (n_sub=1) =====\n" + "="*80)
    key, stage_key = jax.random.split(key)
    current_model  = FBPINN_PoU(
                        key          = stage_key,
                        num_subdomains = 1,
                        mlp_config   = config["mlp_conf"],
                    )

    current_model, loss_hist = train_fbpinn_no_batching(
                                stage_key,
                                current_model,
                                problem,
                                colloc_train,          # ← 全量点
                                config["FBPINN_LR"],
                                config["FBPINN_STEPS"]
                            )
    current_n_sub = 1
    print(f"\nTraining for n_sub=1 complete.  Loss: {loss_hist[-1]:.4e}")

    # ------------------------------------------------------------------ #
    #  在同一规则网格上评估 PINN → 与 U_fdm 对齐                          #
    # ------------------------------------------------------------------ #
    print("\n--- Evaluating Stage 0 (n_sub=1) Error ---")
    u_pinn_flat = jax.vmap(problem.solution_ansatz, in_axes=(None, 0))(
        current_model, fdm_interior_points_jax
    )
    l1_error     = jnp.mean(jnp.abs(u_pinn_flat - u_fdm_flat_jax))
    rel_l1_error = l1_error / jnp.mean(jnp.abs(u_fdm_flat_jax))
    print(f" -> L1 Error: {l1_error:.4e} | Relative L1 Error: {rel_l1_error:.4e}")
    l1_errors_history[current_n_sub] = (float(l1_error), float(rel_l1_error))

    # ------------------------------------------------------------------ #
    #  把预测结果填回规则网格再绘图                                       #
    # ------------------------------------------------------------------ #
    U_pinn_grid               = np.full_like(U_fdm, np.nan)
    U_pinn_grid[interior_mask] = np.asarray(u_pinn_flat)

    plot_results_on_grid(problem, current_model, 0, current_n_sub, save_dir)
    plot_error_map(U_pinn_grid, U_fdm, X_fdm, Y_fdm,
                   l1_error, 0, current_n_sub, save_dir)
    plot_loss_history(loss_hist, 0, current_n_sub, save_dir)
    save_npz_history(save_dir, 0, current_n_sub, loss_hist,
                     l1_error, rel_l1_error)
    
    pou_schedule          = config.get("pou_schedule", [4, 9, 16, 25])
    discovery_start_index = 0          # 指向下一个待尝试的 n_sub
    outer_loop_iter       = 1          # Stage 计数器（Stage 0 已完成）
    while True:
        print("\n" + "#"*80 +
            f"\n##### Iteration {outer_loop_iter}: Discovering Partitions... #####\n"
            + "#"*80)

        if discovery_start_index >= len(pou_schedule):
            print("\nAll PoU schemes attempted. Stopping."); break

        best_n_sub      = None          # 最近一次通过健康检查的 n_sub
        best_window_fn  = None
        # -----------------------------------------------------------
        # ① 逐个尝试 pou_schedule，从 discovery_start_index 开始
        # -----------------------------------------------------------
        for i in range(discovery_start_index, len(pou_schedule)):
            n_sub_try = pou_schedule[i]
            nx, ny    = get_grid_dims(n_sub_try)

            print(f"\n-- Attempting to learn {n_sub_try} partitions "
                f"({nx}×{ny}) --")

            # （不能整列成矩阵的直接跳过）
            if nx * ny != n_sub_try:
                print(f"   -> Skip {n_sub_try}: cannot arrange {nx}×{ny}.")
                continue

            # ---- PoU‑LSGD ------------------------------------------------
            key, pou_key = jax.random.split(key)
            sep_conf = config["sep_mlp_pou_conf_map"].get(
                        n_sub_try, config["default_sep_mlp_pou_conf"])
            pou_net   = SepMLPPOUNet(nx, ny, key=pou_key, **sep_conf)
            lsgd_cfg  = config["lsgd_conf_map"].get(
                        n_sub_try, config["default_lsgd_conf"])

            y_train_pou = jax.vmap(problem.solution_ansatz,
                                in_axes=(None, 0))(current_model, x_test)
            final_pou_params = run_lsgd(
                                pou_net, pou_net.init_params(),
                                x_test, y_train_pou, lsgd_cfg)
            window_fn = WindowModule(pou_net=pou_net, params=final_pou_params)

            # ---- 画 PoU 并做健康检查 ------------------------------------
            plot_pou_on_grid(window_fn, outer_loop_iter, n_sub_try, save_dir)
            ok = check_territory(window_fn, x_test)
            if ok:
                # 记录“最近一次”通过的
                best_n_sub     = n_sub_try
                best_window_fn = window_fn
                discovery_start_index = i + 1      # 指向下一档
                print("   -> PoU healthy, keep searching deeper...\n")
                continue        # 继续尝试更细的分区
            else:
                print("   -> PoU collapsed; stop search this round.\n")
                break           # 立刻停止扫描，回退到 best_n_sub

        # -----------------------------------------------------------
        # ② 若没有任何健康 PoU，则整个算法终止
        # -----------------------------------------------------------
        if best_n_sub is None:
            print("No viable PoU found in remaining schedule. Terminating.")
            break

        # -----------------------------------------------------------
        # ③ 用 best_n_sub / best_window_fn 训练 FBPINN
        # -----------------------------------------------------------
        print("\n" + "="*80 +
            f"\n===== Stage {outer_loop_iter}: FBPINN Training "
            f"(n_sub={best_n_sub}) =====\n" + "="*80)

        key, stage_key = jax.random.split(key)
        current_n_sub  = best_n_sub
        current_model  = FBPINN_PoU(
                            key=stage_key,
                            num_subdomains=current_n_sub,
                            mlp_config=config['mlp_conf'],
                            window_fn=best_window_fn)

        current_model, loss_hist = train_fbpinn_no_batching(
                                    stage_key, current_model, problem,
                                    colloc_train,
                                    config["FBPINN_LR"],
                                    config["FBPINN_STEPS"])

        # ... 后面误差评估/画图/记录保持原逻辑 ...
        outer_loop_iter += 1

        print(f"\nTraining for n_sub={current_n_sub} complete.  Loss: {loss_hist[-1]:.4e}")

        print(f"\n--- Evaluating Stage {outer_loop_iter} (n_sub={current_n_sub}) Error ---")
        u_pinn_flat = jax.vmap(problem.solution_ansatz, in_axes=(None, 0))(current_model, fdm_interior_points_jax)
        l1_error = jnp.mean(jnp.abs(u_pinn_flat - u_fdm_flat_jax))
        rel_l1_error = l1_error / jnp.mean(jnp.abs(u_fdm_flat_jax))
        print(f" -> L1 Error: {l1_error:.4e} | Relative L1 Error: {rel_l1_error:.4e}")
        l1_errors_history[current_n_sub] = (float(l1_error), float(rel_l1_error))

        U_pinn_grid = np.full_like(U_fdm, np.nan)
        U_pinn_grid[np.isfinite(U_fdm)] = np.asarray(u_pinn_flat)

        plot_results_on_grid(problem, current_model, outer_loop_iter, current_n_sub, save_dir)
        plot_error_map(U_pinn_grid, U_fdm, X_fdm, Y_fdm, l1_error, outer_loop_iter, current_n_sub, save_dir)
        plot_loss_history(loss_hist, outer_loop_iter, current_n_sub, save_dir)
        save_npz_history(save_dir, outer_loop_iter, current_n_sub, loss_hist, l1_error, rel_l1_error)
        outer_loop_iter += 1

    print("\n" + "="*80 + "\n=====  L1 Error Summary =====\n" + "="*80)
    sorted_keys = sorted(l1_errors_history.keys())
    n_subs = sorted_keys
    errors = [l1_errors_history[k][0] for k in sorted_keys]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(n_subs, errors, 'o-', label='L1 Error vs. FEM')
    ax.set_xlabel('Number of Subdomains (N)')
    ax.set_ylabel('L1 Error')
    ax.set_title('L1 Error vs. Number of Subdomains')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks(n_subs); ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, which="both", ls="--"); ax.legend()
    plt.tight_layout()
    filepath = os.path.join(save_dir, "l1_error_vs_n_sub.png"); plt.savefig(filepath, dpi=300); plt.close(fig)
    print(f"Summary plot of L1 error vs. N_sub saved to {filepath}")


if __name__ == '__main__':
    config = {
        "FBPINN_STEPS": 20000, "FBPINN_LR": 1e-3, "BATCH_SIZE": 10000,
        "test_n_points": 5000, "colloc_n_points": 10000, "fem_n_refine":40,
        "mlp_conf": dict(in_size=2, out_size=1, width_size=16, depth=2, activation=jnp.tanh),
        "lsgd_conf_map": {
        4:  LSGDConfig(n_epochs=5000, lr=1e-4,  lam_init=0.01, rho=0.99, n_stag=300),
        9:  LSGDConfig(n_epochs=5000, lr=1e-4,  lam_init=0.01, rho=0.99, n_stag=300),
        16:  LSGDConfig(n_epochs=5000, lr=1e-4,  lam_init=0.01, rho=0.99, n_stag=300),
        25:  LSGDConfig(n_epochs=5000, lr=1e-4,  lam_init=0.01, rho=0.99, n_stag=300),

        # 可以继续往下加 16, 25 ……
        },
        "default_lsgd_conf": LSGDConfig(n_epochs=5000, lr=1e-4, n_stag=300),
        
        "sep_mlp_pou_conf_map": {
        4: dict(hidden=(16, 16), tau=1.0),          # 4 个窗口
        9: dict(hidden=(16, 16), tau=1.0),          # 9 个窗口
        16: dict(hidden=(16, 16), tau=1.0),   # 按需继续加
        25: dict(hidden=(16, 16), tau=1.0),   # 按需继续加
        },
        "default_sep_mlp_pou_conf": dict(hidden=(16, 16), tau=2.0),
        "pou_schedule": [4, 9, 16, 25],
    }

    problem_class = PoissonLshapeComplicatedRHS
    print(f"\nSolving problem: {problem_class.__name__}")

    key = jax.random.PRNGKey(42)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config["save_dir"] = f"results_{problem_class.__name__}_{timestamp}"
    os.makedirs(config["save_dir"], exist_ok=True)
    print(f"Results will be saved to: {config['save_dir']}\n")

    run_hierarchical_fbpinn(key, problem_class, config)

    print("\n\n" + "#"*80 + "\n##### Algorithm execution finished. #####\n" + "#"*80)
    print(f"All results and plots have been saved to: '{config['save_dir']}'")
