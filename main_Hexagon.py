# -*- coding: utf-8 -*-
"""
Hierarchical FBPINN with a choice of PoU networks (Separable vs. Standard MLP).
"""

import os
import dataclasses
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# JAX and related libraries
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from jax import config

# Scientific computing and plotting
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D

# FDM reference solver dependencies
import scipy.sparse as sp
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# Progress bar
from tqdm import trange

# --- JAX Configuration ---
config.update("jax_enable_x64", True)


# geometry and ADF for hexagon
@dataclasses.dataclass(frozen=True)
class HexagonGeometry:
    radius: float = 1.0
    phase_shift: float = jnp.pi / 2
    @property
    def vertices(self) -> jax.Array:
        thetas = jnp.linspace(0, 2 * jnp.pi, 7)[:-1]
        return jnp.array([[self.radius * jnp.cos(t + self.phase_shift), self.radius * jnp.sin(t + self.phase_shift)] for t in thetas])
    @property
    def segments(self) -> List[Dict[str, jax.Array]]:
        verts = self.vertices
        return [{"p1": verts[i], "p2": verts[(i + 1) % 6]} for i in range(6)]
    def is_inside(self, xy: jax.Array) -> jax.Array:
        points = jnp.atleast_2d(xy)
        x, y = points[:, 0], points[:, 1]
        is_inside_flags = jnp.ones(x.shape[0], dtype=bool)
        for seg in self.segments:
            p1, p2 = seg['p1'], seg['p2']
            cross_product = (p2[0] - p1[0]) * (y - p1[1]) - (p2[1] - p1[1]) * (x - p1[0])
            is_inside_flags &= (cross_product >= -1e-9)
        return is_inside_flags

def adf_hexagon(point: jax.Array, geometry: HexagonGeometry, m_param: float = 1.0) -> jax.Array:
    @jax.jit
    def _get_phi_for_segment(p: jax.Array, p1: jax.Array, p2: jax.Array) -> jax.Array:
        (x, y), (x1, y1), (x2, y2) = p, p1, p2
        dx_seg, dy_seg = x2 - x1, y2 - y1
        L_sq = dx_seg**2 + dy_seg**2
        L = jnp.sqrt(L_sq)
        L = jnp.where(L < 1e-9, 1.0, L)
        f = ((x - x1) * dy_seg - (y - y1) * dx_seg) / L
        xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
        t = (1 / L) * ((L / 2)**2 - ((x - xc)**2 + (y - yc)**2))
        varphi = jnp.sqrt(t**2 + f**4)
        phi_val = jnp.sqrt(f**2 + ((varphi - t) / 2)**2)
        euclidean_dist = jnp.sqrt((x - x1)**2 + (y - y1)**2)
        return jnp.where(L_sq < 1e-12, euclidean_dist, phi_val)
    @jax.jit
    def _r_intersection(phis: jax.Array) -> jax.Array:
        is_on_boundary = jnp.any(phis < 1e-9)
        safe_phis = jnp.maximum(phis, 1e-12)
        sum_inv_phi_m = jnp.sum(safe_phis**(-m_param))
        combined_phi = jnp.maximum(sum_inv_phi_m, 1e-12)**(-1.0 / m_param)
        return jnp.where(is_on_boundary, 0.0, combined_phi)
    phi_values = jnp.array([_get_phi_for_segment(point, seg['p1'], seg['p2']) for seg in geometry.segments])
    return _r_intersection(phi_values)

#  FDM REFERENCE SOLVER
def solve_poisson_fdm_hexagon(grid_res: int, is_inside_fn: Callable, rhs_fn_np: Callable) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print(f"[FDM] Starting reference solution generation with grid resolution {grid_res}x{grid_res}...")
    xs = np.linspace(-1.0, 1.0, grid_res)
    ys = np.linspace(-1.0, 1.0, grid_res)
    X, Y = np.meshgrid(xs, ys)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1)
    h = 2.0 / (grid_res - 1)
    inside_mask_flat = np.array(is_inside_fn(jnp.array(grid_points)))
    inside_mask_grid = inside_mask_flat.reshape(grid_res, grid_res)
    idx_map = -np.ones((grid_res, grid_res), dtype=int)
    interior_indices = np.where(inside_mask_grid)
    num_interior = len(interior_indices[0])
    idx_map[interior_indices] = np.arange(num_interior)
    print(f"[FDM] Found {num_interior} interior grid nodes.")
    if num_interior == 0:
        raise ValueError("No interior nodes found. Increase grid_res or check geometry.")
    A = lil_matrix((num_interior, num_interior), dtype=float)
    b = np.zeros(num_interior, dtype=float)
    for k, (i, j) in enumerate(zip(*interior_indices)):
        A[k, k] = 4.0
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < grid_res and 0 <= nj < grid_res and inside_mask_grid[ni, nj]:
                neighbor_idx = idx_map[ni, nj]
                A[k, neighbor_idx] = -1.0
        b[k] = (h**2) * rhs_fn_np(X[i, j], Y[i, j])
    print("[FDM] Solving sparse linear system...")
    A_csr = csr_matrix(A)
    u_vec = spsolve(A_csr, b)
    print("[FDM] Solve complete.")
    U_grid = np.full((grid_res, grid_res), np.nan, dtype=float)
    U_grid[interior_indices] = u_vec
    interior_points = np.column_stack([X[interior_indices], Y[interior_indices]])
    return U_grid, X, Y, interior_points, u_vec

#  PDE DEFINITION
class PoissonProblem(eqx.Module):
    adf_fn: Callable
    domain: List[List[float]] = eqx.static_field()
    def __init__(self, adf_fn: Callable, domain_bounds: List[List[float]]):
        self.adf_fn = adf_fn
        self.domain = domain_bounds
    def solution_ansatz(self, model: Callable, x: jax.Array) -> jax.Array:
        return self.adf_fn(x) * model(x).squeeze()
    def rhs_f(self, x_in: jax.Array) -> jax.Array:
        raise NotImplementedError
    def pointwise_residual(self, model: Callable, x: jax.Array) -> jax.Array:
        u_hessian = jax.hessian(lambda pt: self.solution_ansatz(model, pt))(x)
        laplacian_u = jnp.trace(u_hessian)
        return -laplacian_u - self.rhs_f(x)
    def residual_loss(self, model: Callable, xy: jax.Array) -> jax.Array:
        if xy.shape[0] == 0:
            return 0.0
        residuals = jax.vmap(self.pointwise_residual, in_axes=(None, 0))(model, xy)
        return jnp.mean(residuals**2)

class PoissonHexagonComplicatedRHS(PoissonProblem):
    k_min: float = 2.0
    k_max: float = 10.0
    omega: float = jnp.pi
    p: float = 2.0
    def rhs_f(self, x_in: jax.Array) -> jax.Array:
        x, y = x_in[0], x_in[1]
        kx = self.k_min + (self.k_max - self.k_min) * ((x + 1) / 2)**self.p
        ky = self.k_min + (self.k_max - self.k_min) * ((y + 1) / 2)**self.p
        return 100 * (jnp.sin(kx * self.omega * x) + jnp.sin(ky * self.omega * y))

#  NEURAL NETWORK & PARTITION OF UNITY, the FBPINN here is not the same as the one in model/fbpinn_model.py which is more adapted to complex geometries
class CustomMLP(eqx.Module):
    layers: List[eqx.nn.Linear]
    activation: Callable = eqx.static_field()
    final_activation: Callable = eqx.static_field()
    def __init__(self, *, in_size: int, out_size: int, width_size: int, depth: int, activation: Callable, key: jax.Array, final_activation: Callable = lambda x: x):
        self.activation = activation
        self.final_activation = final_activation
        keys = jax.random.split(key, depth + 1)
        self.layers = []
        if depth == 0:
            self.layers.append(eqx.nn.Linear(in_size, out_size, key=keys[0]))
        else:
            self.layers.append(eqx.nn.Linear(in_size, width_size, key=keys[0]))
            for i in range(depth - 1):
                self.layers.append(eqx.nn.Linear(width_size, width_size, key=keys[i+1]))
            self.layers.append(eqx.nn.Linear(width_size, out_size, key=keys[depth]))
    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.final_activation(self.layers[-1](x))

class FBPINN(eqx.Module):
    subnets: List[CustomMLP]
    window_fn: Optional[Callable] = eqx.static_field()
    def __init__(self, key: jax.Array, num_subdomains: int, mlp_config: Dict, window_fn: Optional[Callable] = None):
        keys = jax.random.split(key, num_subdomains)
        self.subnets = [CustomMLP(**mlp_config, key=k) for k in keys]
        self.window_fn = window_fn
    def __call__(self, x: jax.Array) -> jax.Array:
        points = jnp.atleast_2d(x)
        if len(self.subnets) == 1 or self.window_fn is None:
            return jax.vmap(self.subnets[0])(points).squeeze()
        partitions = self.window_fn(points)
        subnet_outputs = jnp.stack([jax.vmap(net)(points) for net in self.subnets], axis=-1).squeeze()
        return jnp.sum(partitions * subnet_outputs, axis=1)

# --- PoU Networks ---
def _glorot_uniform(key, shape):
    fan_in, fan_out = shape[0], shape[1]
    lim = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-lim, maxval=+lim)

class MLPPOUNet:
    """A standard, non-separable MLP for Partition of Unity."""
    def __init__(self, input_dim: int, num_experts: int, hidden: Sequence[int] = (64, 64), key: Optional[jax.Array] = None):
        self.input_dim, self.num_experts = int(input_dim), int(num_experts)
        key = jax.random.PRNGKey(42) if key is None else key
        keys = jax.random.split(key, len(hidden) + 1)
        p, in_dim = {}, self.input_dim
        for i, h in enumerate(hidden):
            p[f"W{i}"] = _glorot_uniform(keys[i], (in_dim, h))
            p[f"b{i}"] = jnp.zeros((h,))
            in_dim = h
        p["W_out"] = _glorot_uniform(keys[-1], (in_dim, self.num_experts))
        p["b_out"] = jnp.zeros((self.num_experts,))
        self._init_params = p
    def init_params(self) -> Dict[str, Any]:
        return {k: v.copy() for k, v in self._init_params.items()}
    @staticmethod
    def forward(params: Dict[str, Any], x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.atleast_2d(x)
        h = x
        n_layer = (len(params) // 2) - 1
        for i in range(n_layer):
            h = jax.nn.relu(h @ params[f"W{i}"] + params[f"b{i}"])
        logits = h @ params["W_out"] + params["b_out"]
        return jax.nn.softmax(logits, axis=-1)

def _init_mlp_1d(key, hidden: Sequence[int], out_dim: int) -> Dict[str, Any]:
    dims = [1] + list(hidden) + [out_dim]
    n_lay = len(dims) - 1
    keys = jax.random.split(key, n_lay)
    params = {}
    for i, (m, n) in enumerate(zip(dims[:-1], dims[1:])):
        params[f"W{i}"] = _glorot_uniform(keys[i], (m, n))
        params[f"b{i}"] = 1e-2 * jax.random.normal(keys[i], (n,))
    return params

def _mlp_forward_1d(params: Dict[str, Any], x: jnp.ndarray) -> jnp.ndarray:
    if x.ndim == 1: x = x[:, None]
    h = x
    L = (len(params) // 2) - 1
    for i in range(L):
        h = jnp.tanh(h @ params[f"W{i}"] + params[f"b{i}"])
    return h @ params[f"W{L}"] + params[f"b{L}"]

class SepMLPPOUNet:
    """A Separable MLP-based Partition of Unity network."""
    def __init__(self, nx: int, ny: int, hidden: Sequence[int]=(32,32), tau: float=0.1, key: jax.random.PRNGKey=jax.random.PRNGKey(0)):
        self.nx, self.ny = int(nx), int(ny)
        self.num_experts = self.nx * self.ny
        self.tau = float(tau)
        kx, ky = jax.random.split(key)
        self.param_x = _init_mlp_1d(kx, hidden, self.nx)
        self.param_y = _init_mlp_1d(ky, hidden, self.ny)
    def init_params(self) -> Dict[str, Any]:
        return {"x": self.param_x, "y": self.param_y}
    def forward(self, params: Dict[str, Any], xy: jnp.ndarray) -> jnp.ndarray:
        xy = jnp.atleast_2d(xy)
        x, y = xy[:, :1], xy[:, 1:]
        z_x = _mlp_forward_1d(params["x"], x)
        z_y = _mlp_forward_1d(params["y"], y)
        logits = (z_x[:, :, None] + z_y[:, None, :]) / self.tau
        logits = logits.reshape(xy.shape[0], -1)
        logits = logits - jnp.max(logits, axis=1, keepdims=True)
        return jax.nn.softmax(logits, axis=1)

class WindowModule(eqx.Module):
    pou_net: Any = eqx.static_field()
    params: Dict
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.pou_net.forward(self.params, x)

@dataclasses.dataclass
class LSGDConfig:
    n_epochs: int = 5000
    lr: float = 1e-4

def _design_matrix(x: jnp.ndarray, degree: int) -> jnp.ndarray:
    if x.ndim == 1: x = x.reshape(-1, 2)
    x1, x2 = x[:, 0], x[:, 1]
    features = [jnp.ones_like(x1)]
    if degree > 0:
        for d in range(1, degree + 1):
            for i in range(d + 1):
                j = d - i
                features.append((x1**i) * (x2**j))
    return jnp.stack(features, axis=1)

def fit_local_polynomials(x, y, w):
    poly_degree = 2
    A = _design_matrix(x, poly_degree)
    y_col = y.reshape(-1, 1)
    def _solve(weights, A_matrix, y_vector):
        Aw = A_matrix * weights.reshape(-1, 1)
        M = A_matrix.T @ Aw
        b = (Aw.T @ y_vector).squeeze()
        coeffs = jnp.linalg.pinv(M) @ b
        return coeffs
    return jax.vmap(_solve, in_axes=(1, None, None), out_axes=0)(w, A, y_col)

def _predict_from_coeffs(x, coeffs, partitions, degree=2):
    A = _design_matrix(x, degree)
    y_cent = A @ coeffs.T
    return jnp.sum(partitions * y_cent, 1)

def run_lsgd(pou_net, initial_params, x, y, cfg: LSGDConfig):
    params = initial_params
    def loss_fn(p):
        part = pou_net.forward(p, x)
        coeff = fit_local_polynomials(x, y, part)
        pred = _predict_from_coeffs(x, coeff, part)
        return jnp.mean((pred - y)**2)
    valgrad_fn = jax.jit(jax.value_and_grad(loss_fn))
    opt = optax.adam(cfg.lr)
    opt_state = opt.init(params)
    bar = trange(cfg.n_epochs, desc=f"PoU-LSGD (N={pou_net.num_experts})", dynamic_ncols=True)
    for ep in bar:
        loss_val, grads = valgrad_fn(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)
        bar.set_postfix(loss=f"{loss_val:.3e}")
    return params


#  TRAINING & SAMPLING
@dataclasses.dataclass
class TrainerConfig:
    learning_rate: float = 1e-3
    steps: int = 20000
    batch_size: Optional[int] = None
@dataclasses.dataclass
class RADConfig:
    enabled: bool = True
    refresh_every: int = 5000
    colloc_size: int = 10000
    pool_size: int = 40000
    oversample_factor: int = 3
    k: float = 3.0
    c: float = 1.0

def generate_grid_collocation(n_points: int, domain_bounds: List, is_inside_fn: Callable) -> jax.Array:
    dom_min, dom_max = np.array(domain_bounds[0]), np.array(domain_bounds[1])
    n_side = int(np.ceil(np.sqrt(n_points)))
    xs = np.linspace(dom_min[0], dom_max[0], n_side)
    ys = np.linspace(dom_min[1], dom_max[1], n_side)
    X, Y = np.meshgrid(xs, ys)
    grid_points = jnp.array(np.stack([X.ravel(), Y.ravel()], axis=-1))
    inside_mask = is_inside_fn(grid_points)
    inside_pts = grid_points[inside_mask]
    print(f"[Grid Sampling] Generated {grid_points.shape[0]} candidate points, kept {inside_pts.shape[0]} interior points.")
    return inside_pts

def rad_sampler(key: jax.Array, model: FBPINN, problem: PoissonProblem, config: RADConfig, is_inside_fn: Callable) -> Tuple[jax.Array, jax.Array]:
    lo, hi = jnp.array(problem.domain[0]), jnp.array(problem.domain[1])
    dim = lo.size
    key, subkey = jax.random.split(key)
    pool_candidates = jax.random.uniform(subkey, (config.pool_size * config.oversample_factor, dim), minval=lo, maxval=hi)
    pool = pool_candidates[is_inside_fn(pool_candidates)][:config.pool_size]
    params, static = eqx.partition(model, eqx.is_array)
    residuals = jax.vmap(problem.pointwise_residual, (None, 0))(eqx.combine(params, static), pool)
    abs_residuals = jnp.abs(residuals)
    weights = (abs_residuals**config.k) / (jnp.mean(abs_residuals**config.k) + 1e-9) + config.c
    probs = weights / jnp.sum(weights)
    key, subkey = jax.random.split(key)
    indices = jax.random.choice(subkey, pool.shape[0], shape=(config.colloc_size,), p=probs, replace=False)
    return key, pool[indices]

def train_model(key: jax.Array, model: FBPINN, problem: PoissonProblem, train_cfg: TrainerConfig, rad_cfg: Optional[RADConfig], initial_collocation: jax.Array, is_inside_fn: Callable) -> Tuple[FBPINN, jax.Array]:
    params, static = eqx.partition(model, eqx.is_array)
    optimizer = optax.adam(train_cfg.learning_rate)
    opt_state = optimizer.init(params)
    collocation_points = initial_collocation
    loss_history = []
    @eqx.filter_jit
    def step(p, o, batch):
        loss, grads = jax.value_and_grad(problem.residual_loss)(eqx.combine(p, static), batch)
        updates, new_o = optimizer.update(grads, o, p)
        new_p = eqx.apply_updates(p, updates)
        return new_p, new_o, loss
    bar_desc = f"FBPINN (N={len(model.subnets)})"
    if rad_cfg and rad_cfg.enabled:
        bar_desc += " + RAD"
    bar = trange(train_cfg.steps, desc=bar_desc, dynamic_ncols=True)
    for s in bar:
        if rad_cfg and rad_cfg.enabled and s > 0 and s % rad_cfg.refresh_every == 0:
            key, collocation_points = rad_sampler(key, eqx.combine(params, static), problem, rad_cfg, is_inside_fn)
        if train_cfg.batch_size is None:
            batch = collocation_points
        else:
            key, subkey = jax.random.split(key)
            idx = jax.random.choice(subkey, collocation_points.shape[0], (train_cfg.batch_size,), replace=False)
            batch = collocation_points[idx]
        params, opt_state, loss_val = step(params, opt_state, batch)
        if jnp.isnan(loss_val) or jnp.isinf(loss_val):
            print(f"\nError: Loss became NaN/Inf at step {s}. Aborting training.")
            break
        loss_history.append(float(loss_val))
        bar.set_postfix(loss=f"{loss_val:.3e}")
    return eqx.combine(params, static), jnp.array(loss_history)

#  UTILITIES (Metrics, Plotting, Health Check)
def compute_error_metrics(pred: jax.Array, true: jax.Array) -> Dict[str, float]:
    err = pred - true
    metrics = {"MAE": float(jnp.mean(jnp.abs(err))), "MSE": float(jnp.mean(err**2)), "RMSE": float(jnp.sqrt(jnp.mean(err**2))), "RelL2": float(jnp.linalg.norm(err) / (jnp.linalg.norm(true) + 1e-12))}
    return metrics

def get_grid_dims(n_sub: int) -> Tuple[int, int]:
    if n_sub == 1: return 1, 1
    if n_sub in [2, 3, 5, 7, 11, 13]: return 1, n_sub
    best_factor = 1
    for i in range(2, int(jnp.sqrt(n_sub)) + 1):
        if n_sub % i == 0: best_factor = i
    return (best_factor, n_sub // best_factor) if best_factor != 1 else (1, n_sub)

def check_territory(window_fn: Callable, x_test: jax.Array, dominance_threshold: float = 0.8) -> bool:
    weights = window_fn(x_test)
    num_partitions = weights.shape[1]
    if num_partitions == 1: return True
    dominant_indices = jnp.argmax(weights, axis=1)
    unique_dominant = jnp.unique(dominant_indices)
    if len(unique_dominant) < num_partitions:
        missing = jnp.setdiff1d(jnp.arange(num_partitions), unique_dominant)
        print(f"    - PoU Health Check FAILED: Partitions {missing} have no territory!")
        return False
    for i in range(num_partitions):
        territory_weights = weights[dominant_indices == i, i]
        if territory_weights.size > 0 and jnp.max(territory_weights) < dominance_threshold:
            print(f"    - PoU Health Check FAILED: Partition {i} is too weak (peak weight {jnp.max(territory_weights):.3e}).")
            return False
    print(f"    - PoU Health Check PASSED for {num_partitions} partitions.")
    return True

def plot_solution(u_grid: np.ndarray, title: str, save_path: str, cmap: str = 'viridis'):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(u_grid, extent=[-1, 1, -1, 1], origin='lower', cmap=cmap)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_loss_history(loss_hist: np.ndarray, stage: int, n_sub: int, save_dir: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(loss_hist)
    ax.set_yscale('log')
    ax.set_title(f'Stage {stage}: Loss History (N={n_sub})')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Log Loss (PDE Residual)')
    ax.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"loss_stage_{stage}_nsub_{n_sub}.png"), dpi=300)
    plt.close(fig)

def save_metrics_history(save_dir, stage, n_sub, loss, metrics):
    path = os.path.join(save_dir, f"history_stage_{stage}_nsub_{n_sub}.npz")
    np.savez_compressed(path, loss_hist=np.asarray(loss), **metrics)

# main experiment function
def run_experiment(config: Dict):
    key = jax.random.PRNGKey(config["seed"])
    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Setup Geometry and PDE Problem
    geometry = HexagonGeometry()
    adf_func = lambda p: adf_hexagon(p, geometry)
    problem = PoissonHexagonComplicatedRHS(adf_fn=adf_func, domain_bounds=[[-geometry.radius, -geometry.radius], [geometry.radius, geometry.radius]])
    
    # 2. Generate FDM Reference Solution
    def rhs_numpy(x, y):
        p_rhs = problem.p
        kx = problem.k_min + (problem.k_max - problem.k_min) * ((x + 1) / 2)**p_rhs
        ky = problem.k_min + (problem.k_max - problem.k_min) * ((y + 1) / 2)**p_rhs
        return 100 * (np.sin(kx * problem.omega * x) + np.sin(ky * problem.omega * y))
    U_ref, X_ref, Y_ref, ref_points, u_ref_flat = solve_poisson_fdm_hexagon(grid_res=config["fdm_grid_res"], is_inside_fn=geometry.is_inside, rhs_fn_np=rhs_numpy)
    ref_points_jax, u_ref_flat_jax = jnp.array(ref_points), jnp.array(u_ref_flat)
    plot_solution(U_ref, "FDM Reference Solution", os.path.join(save_dir, "reference_fdm.png"))
    
    # 3. Generate initial collocation points
    colloc_train = generate_grid_collocation(config["colloc_n_points"], problem.domain, geometry.is_inside)
    x_test = generate_grid_collocation(config["test_n_points"], problem.domain, geometry.is_inside)
    
    metrics_history = {}
    
    # 4. Stage 0: Train base PINN (N=1)
    print("\n" + "="*80 + "\n===== Stage 0: Training Base PINN (N=1) =====\n" + "="*80)
    key, stage_key = jax.random.split(key)
    current_model = FBPINN(key=stage_key, num_subdomains=1, mlp_config=config["mlp_conf"])
    current_model, loss_hist = train_model(stage_key, current_model, problem, config["trainer_config"], config.get("rad_config"), colloc_train, geometry.is_inside)
    
    print("\n--- Evaluating Stage 0 (N=1) ---")
    u_pred_flat = jax.vmap(problem.solution_ansatz, (None, 0))(current_model, ref_points_jax)
    metrics = compute_error_metrics(u_pred_flat, u_ref_flat_jax)
    metrics_history[1] = metrics
    print(f" -> RMSE: {metrics['RMSE']:.4e} | RelL2: {metrics['RelL2']:.4e}")
    
    U_pred_grid = np.full_like(U_ref, np.nan)
    U_pred_grid[np.isfinite(U_ref)] = np.asarray(u_pred_flat)
    plot_solution(np.abs(U_pred_grid - U_ref), f"Stage 0 Error (RMSE={metrics['RMSE']:.2e})", os.path.join(save_dir, "error_stage_0_nsub_1.png"), cmap='plasma')
    plot_loss_history(loss_hist, 0, 1, save_dir)
    save_metrics_history(save_dir, 0, 1, loss_hist, metrics)
    
    # 5. Hierarchical Training Loop
    pou_schedule = config.get("pou_schedule", [4, 9, 16, 25])
    discovery_start_index = 0
    outer_loop_iter = 1
    
    while discovery_start_index < len(pou_schedule):
        print("\n" + "#"*80 + f"\n##### Iteration {outer_loop_iter}: Discovering Partitions... #####\n" + "#"*80)
        best_n_sub, best_window_fn = None, None
        
        for i in range(discovery_start_index, len(pou_schedule)):
            n_sub_try = pou_schedule[i]
            nx, ny = get_grid_dims(n_sub_try)
            print(f"\n-- Attempting to learn {n_sub_try} partitions ({nx}x{ny}) --")
            
            key, pou_key = jax.random.split(key)
            pou_backend = config.get("pou_backend", "sep_mlp")
            print(f"--> Using PoU Backend: '{pou_backend}'")

            if pou_backend == "sep_mlp":
                pou_net = SepMLPPOUNet(nx, ny, key=pou_key, **config["sep_mlp_pou_conf"])
            elif pou_backend == "mlp":
                pou_net = MLPPOUNet(input_dim=2, num_experts=n_sub_try, key=pou_key, **config["mlp_pou_conf"])
            else:
                raise ValueError(f"Unknown pou_backend: {pou_backend}")
            
            y_train_pou = jax.vmap(problem.solution_ansatz, (None, 0))(current_model, x_test)
            final_pou_params = run_lsgd(pou_net, pou_net.init_params(), x_test, y_train_pou, config["lsgd_config"])
            window_fn = WindowModule(pou_net=pou_net, params=final_pou_params)
            
            if check_territory(window_fn, x_test):
                best_n_sub, best_window_fn = n_sub_try, window_fn
                discovery_start_index = i + 1
                continue
            else:
                break
        
        if best_n_sub is None:
            print("No viable PoU found in the remaining schedule. Terminating.")
            break
            
        print("\n" + "="*80 + f"\n===== Stage {outer_loop_iter}: Training FBPINN (N={best_n_sub}) =====\n" + "="*80)
        
        key, stage_key = jax.random.split(key)
        current_model = FBPINN(key=stage_key, num_subdomains=best_n_sub, mlp_config=config['mlp_conf'], window_fn=best_window_fn)
        current_model, loss_hist = train_model(stage_key, current_model, problem, config["trainer_config"], config.get("rad_config"), colloc_train, geometry.is_inside)
        
        print(f"\n--- Evaluating Stage {outer_loop_iter} (N={best_n_sub}) ---")
        u_pred_flat = jax.vmap(problem.solution_ansatz, (None, 0))(current_model, ref_points_jax)
        metrics = compute_error_metrics(u_pred_flat, u_ref_flat_jax)
        metrics_history[best_n_sub] = metrics
        print(f" -> RMSE: {metrics['RMSE']:.4e} | RelL2: {metrics['RelL2']:.4e}")
        
        U_pred_grid.fill(np.nan)
        U_pred_grid[np.isfinite(U_ref)] = np.asarray(u_pred_flat)
        plot_solution(np.abs(U_pred_grid - U_ref), f"Stage {outer_loop_iter} Error (RMSE={metrics['RMSE']:.2e})", os.path.join(save_dir, f"error_stage_{outer_loop_iter}_nsub_{best_n_sub}.png"), cmap='plasma')
        plot_loss_history(loss_hist, outer_loop_iter, best_n_sub, save_dir)
        save_metrics_history(save_dir, outer_loop_iter, best_n_sub, loss_hist, metrics)
        outer_loop_iter += 1

    # 6. Final Summary
    print("\n" + "="*80 + "\n===== Error Metric Summary =====\n" + "="*80)
    sorted_keys = sorted(metrics_history.keys())
    n_subs = sorted_keys
    rel_l2_values = [metrics_history[k]["RelL2"] for k in sorted_keys]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(n_subs, rel_l2_values, 'o-', label='Relative L2 vs. FDM')
    ax.set_xlabel('Number of Subdomains (N)')
    ax.set_ylabel('Relative L2 Error')
    ax.set_title('Relative L2 Error vs. Number of Subdomains')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks(n_subs); ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, which="both", ls="--")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "relL2_vs_n_sub_summary.png"), dpi=300)
    plt.close(fig)
    print(f"Summary plot saved to: {save_dir}")


if __name__ == '__main__':
    main_config = {
        "seed": 42,
        "save_dir": f"results/FBPINN_FDM_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        
        "fdm_grid_res": 201,
        "colloc_n_points": 10000,
        "test_n_points": 5000,
        
        "mlp_conf": dict(in_size=2, out_size=1, width_size=16, depth=2, activation=jnp.tanh),
        
        "trainer_config": TrainerConfig(steps=20000, learning_rate=1e-3, batch_size=None),
        
        "rad_config": RADConfig(
            enabled=True,
            refresh_every=10000,
            colloc_size=10000,
            pool_size=20000,
            k=1.0,
            c=1.0
        ),
        
        "lsgd_config": LSGDConfig(n_epochs=5000, lr=1e-4),
        
        "pou_backend": "mlp",  # Choose "mlp" or "sep_mlp"
        
        # Config for the standard MLP PoU ("mlp")
        "mlp_pou_conf": {"hidden": (32, 32)}, 
        
        # Config for the Separable MLP PoU ("sep_mlp")
        "sep_mlp_pou_conf": {"hidden": (16, 16), "tau": 1.0},
        
        "pou_schedule": [4, 9, 16, 25],
    }
    
    run_experiment(main_config)
    
    print("\n\n" + "#"*80 + "\n##### Experiment finished. #####\n" + "#"*80)