from __future__ import annotations
import jax, jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from functools import partial
import os, sys, json, csv
import equinox as eqx
import optax
from typing import Sequence, Dict, Tuple, Any, Callable, Optional
from tqdm import trange
import numpy as np
import dataclasses

project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.append(project_root)

from physics.problems import FirstOrderFreq1010
from model.fbpinn_model import FBPINN_PoU
from utils.data_utils import generate_collocation

class BasePOUNet:
    num_experts: int
    def init_params(self) -> Dict[str, Any]:
        raise NotImplementedError
    def forward(self, params: Dict[str, Any], x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

def glorot(key: jax.random.PRNGKey, shape: tuple[int, int]) -> jnp.ndarray:
    fan_in, fan_out = shape
    lim = jnp.sqrt(6. / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-lim, maxval=lim)

class MLPPOUNet(BasePOUNet):
    def __init__(self, input_dim: int, num_experts: int, hidden: Sequence[int] = (64, 64), key=None):
        self.input_dim, self.num_experts = input_dim, num_experts
        key = jax.random.PRNGKey(42) if key is None else key
        keys = jax.random.split(key, len(hidden) + 1)
        p, in_dim = {}, input_dim
        for i, h in enumerate(hidden):
            p[f"W{i}"] = glorot(keys[i], (in_dim, h))
            p[f"b{i}"] = jnp.zeros((h,))
            in_dim = h
        p["W_out"] = glorot(keys[-1], (in_dim, num_experts))
        p["b_out"] = jnp.zeros((num_experts,))
        self._init_params = p
    def init_params(self) -> Dict[str, Any]:
        return {k: v.copy() for k, v in self._init_params.items()}
    @staticmethod
    def forward(params: Dict[str, Any], x: jnp.ndarray) -> jnp.ndarray:
        h = jnp.atleast_2d(x)
        n_layer = (len(params) // 2) - 1
        for i in range(n_layer):
            h = jax.nn.relu(h @ params[f"W{i}"] + params[f"b{i}"])
        logits = h @ params["W_out"] + params["b_out"]
        return jax.nn.softmax(logits, axis=-1)

class RBFPOUNet(BasePOUNet):
    def __init__(self, input_dim: int, num_centers: int, domain: tuple, key=None):
        self.input_dim, self.num_experts = input_dim, num_centers
        key = jax.random.PRNGKey(42) if key is None else key
        min_b, max_b = jnp.array(domain[0]), jnp.array(domain[1])
        if input_dim == 1:
            xs = jnp.linspace(min_b[0], max_b[0], num_centers)
            base_centers = xs[:, None]
        elif input_dim == 2:
            nx = ny = int(jnp.sqrt(num_centers))
            if nx * ny == num_centers:
                xs = jnp.linspace(min_b[0], max_b[0], nx)
                ys = jnp.linspace(min_b[1], max_b[1], ny)
                grid_x, grid_y = jnp.meshgrid(xs, ys)
                base_centers = jnp.vstack([grid_x.ravel(), grid_y.ravel()]).T
            else:
                base_centers = jax.random.uniform(key, (num_centers, input_dim), minval=min_b, maxval=max_b)
        else:
             base_centers = jax.random.uniform(key, (num_centers, input_dim), minval=min_b, maxval=max_b)
        key, subkey = jax.random.split(key)
        jitters = 0.05 * (max_b - min_b) * jax.random.normal(subkey, base_centers.shape)
        self._init_centers = base_centers + jitters
        self._init_widths = 0.25 * jnp.ones((num_centers,))
    def init_params(self) -> Dict[str, Any]:
        return {"centers": self._init_centers.copy(), "widths": self._init_widths.copy()}
    @staticmethod
    def forward(params: Dict[str, Any], x: jnp.ndarray) -> jnp.ndarray:
        c, w = params["centers"], params["widths"]
        d2 = jnp.sum((x[:, None, :] - c[None, :, :])**2, axis=-1)
        log_phi = -d2 / (w**2 + 1e-12)
        log_phi_stable = log_phi - jnp.max(log_phi, axis=1, keepdims=True)
        phi = jnp.exp(log_phi_stable)
        return phi / jnp.sum(phi, axis=1, keepdims=True)

def _init_mlp_1d(key, hidden: Sequence[int], out_dim: int) -> Dict[str, Any]:
    params = {}
    dims = [1] + list(hidden) + [out_dim]
    keys = jax.random.split(key, len(dims) - 1)
    for i, (m, n) in enumerate(zip(dims[:-1], dims[1:])):
        params[f'W{i}'] = glorot(keys[i], (m, n))
        params[f'b{i}'] = jnp.zeros((n,))
    return params

def _mlp_forward_1d(params: Dict[str, Any], x: jnp.ndarray) -> jnp.ndarray:
    h = jnp.atleast_2d(x)
    num_layers = len(params) // 2
    for i in range(num_layers - 1):
        h = jnp.tanh(h @ params[f'W{i}'] + params[f'b{i}'])
    logits = h @ params[f'W{num_layers - 1}'] + params[f'b{num_layers - 1}']
    return logits

def _rbf_forward_1d(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.atleast_2d(x)
    c, w = params["centers"], params["widths"]
    d2 = (x - c)**2
    log_phi = -d2 / (w**2 + 1e-12)
    return log_phi

class SepMLPPOUNet(BasePOUNet):
    def __init__(self, nx: int, ny: int, hidden: Sequence[int] = (32, 32), tau: float = 1.0, key=None):
        self.nx, self.ny, self.num_experts, self.tau = nx, ny, nx * ny, float(tau)
        key = jax.random.PRNGKey(42) if key is None else key
        kx, ky = jax.random.split(key)
        self._param_x = _init_mlp_1d(kx, hidden, nx)
        self._param_y = _init_mlp_1d(ky, hidden, ny)
    def init_params(self) -> Dict[str, Any]:
        return {
            'x': jax.tree_util.tree_map(lambda x: x.copy(), self._param_x),
            'y': jax.tree_util.tree_map(lambda x: x.copy(), self._param_y)
        }
    def forward(self, params: Dict[str, Any], xy: jnp.ndarray) -> jnp.ndarray:
        xy = jnp.atleast_2d(xy)
        x, y = xy[:, :1], xy[:, 1:]
        z_x = _mlp_forward_1d(params['x'], x)
        z_y = _mlp_forward_1d(params['y'], y)
        logits = (z_x[:, :, None] + z_y[:, None, :]) / self.tau
        logits_flat = logits.reshape(x.shape[0], -1)
        return jax.nn.softmax(logits_flat, axis=-1)

class SepRBFPOUNet(BasePOUNet):
    def __init__(self, nx: int, ny: int, domain: tuple, tau: float = 1.0, key=None):
        self.input_dim, self.nx, self.ny = 2, int(nx), int(ny)
        self.num_experts, self.tau = self.nx * self.ny, float(tau)
        key = jax.random.PRNGKey(42) if key is None else key
        min_b, max_b = jnp.array(domain[0]), jnp.array(domain[1])
        centers_x = jnp.linspace(min_b[0], max_b[0], self.nx)[None, :]
        centers_y = jnp.linspace(min_b[1], max_b[1], self.ny)[None, :]
        width_x = 0.5 * (max_b[0] - min_b[0]) / self.nx
        width_y = 0.5 * (max_b[1] - min_b[1]) / self.ny
        self._param_x = {"centers": centers_x, "widths": jnp.full((self.nx,), width_x)}
        self._param_y = {"centers": centers_y, "widths": jnp.full((self.ny,), width_y)}
    def init_params(self) -> dict:
        return {
            "x": jax.tree_util.tree_map(lambda x: x.copy(), self._param_x),
            "y": jax.tree_util.tree_map(lambda x: x.copy(), self._param_y)
        }
    def forward(self, params: dict, xy: jnp.ndarray) -> jnp.ndarray:
        xy = jnp.atleast_2d(xy)
        x, y = xy[:, :1], xy[:, 1:]
        z_x = _rbf_forward_1d(params["x"], x)
        z_y = _rbf_forward_1d(params["y"], y)
        logits = (z_x[:, :, jnp.newaxis] + z_y[:, jnp.newaxis, :]) / self.tau
        logits_flat = logits.reshape(xy.shape[0], -1)
        return jax.nn.softmax(logits_flat, axis=1)

def _resolve_pou_conf(pou_type: str, config: Dict[str, Any], n_sub: int) -> Dict[str, Any]:
    key_base = f"{pou_type}_pou_conf"
    key_over = f"{pou_type}_pou_overrides"
    base = dict(config.get(key_base, {}))
    over_map = config.get(key_over, {})
    if isinstance(over_map, dict) and n_sub in over_map:
        base.update(over_map[n_sub])
    return base

def make_pou_net(pou_type: str, n_sub: int, domain: Tuple[jnp.ndarray, jnp.ndarray], problem_dim: int, key, config: Dict[str, Any]) -> Tuple[BasePOUNet, Dict[str, Any]]:
    pou_type = pou_type.lower()
    conf = _resolve_pou_conf(pou_type, config, n_sub)
    if pou_type == "sep_mlp":
        nx = int(np.sqrt(n_sub)); ny = n_sub // nx
        if nx * ny != n_sub: raise ValueError(f"SepMLP needs a square number of subdomains, but got n_sub={n_sub}.")
        net = SepMLPPOUNet(nx=nx, ny=ny, **conf, key=key)
    elif pou_type == "sep_rbf":
        nx = int(np.sqrt(n_sub)); ny = n_sub // nx
        if nx * ny != n_sub: raise ValueError(f"SepRBF needs a square number of subdomains, but got n_sub={n_sub}.")
        net = SepRBFPOUNet(nx=nx, ny=ny, domain=domain, **conf, key=key)
    elif pou_type == "rbf":
        net = RBFPOUNet(input_dim=problem_dim, num_centers=n_sub, domain=domain, key=key)
    elif pou_type == "mlp":
        net = MLPPOUNet(input_dim=problem_dim, num_experts=n_sub, **conf, key=key)
    else:
        raise ValueError(f"Unknown pou_type={pou_type}.")
    return net, net.init_params()

# 3. Local Fitting and PoU Training

def _design_matrix(x: jnp.ndarray) -> jnp.ndarray:
    d = x.shape[-1]
    if d == 1:
        x_flat = x.squeeze(-1) if x.ndim > 1 else x
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

@dataclasses.dataclass
class LSGDConfig:
    n_epochs: int = 15000
    lr: float = 1e-4

def _predict_from_coeffs(x, coeffs, partitions):
    A = _design_matrix(x); y_cent = A @ coeffs.T; return jnp.sum(partitions * y_cent, 1)

def run_lsgd(pou_net, initial_params, x, y, cfg: LSGDConfig):
    params = initial_params
    def loss_fn(p):
        part = pou_net.forward(p, x)
        coeffs = fit_local_polynomials(x, y, part, lam=0.0)
        pred = _predict_from_coeffs(x, coeffs, part)
        return jnp.mean((pred - y)**2)
    valgrad_fn = jax.jit(jax.value_and_grad(loss_fn))
    opt = optax.adam(cfg.lr); opt_state = opt.init(params)
    best, best_params = jnp.inf, params
    bar = trange(cfg.n_epochs, desc=f"PoU-LSGD (N={pou_net.num_experts})", dynamic_ncols=True)
    for ep in bar:
        loss_val, grads = valgrad_fn(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if float(loss_val) < float(best) - 1e-12:
            best, best_params = loss_val, params
        bar.set_postfix(loss=f"{float(loss_val):.3e}")
    return best_params


# Metrics & RAD Sampling Functions

def _relative_l2(pred, truth):
    pred = pred.squeeze(); truth = truth.squeeze()
    num = jnp.linalg.norm(pred - truth)
    den = jnp.linalg.norm(truth) + 1e-12
    return num / den
def _mae(pred, truth):
    pred = pred.squeeze(); truth = truth.squeeze()
    return jnp.mean(jnp.abs(pred - truth))
def _mse(pred, truth):
    pred = pred.squeeze(); truth = truth.squeeze()
    return jnp.mean((pred - truth)**2)
def _rmse(pred, truth):
    pred = pred.squeeze(); truth = truth.squeeze()
    return jnp.sqrt(jnp.mean((pred - truth)**2))

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


#  Training Functions
def train_pinn_with_batching(
    key: jax.Array, model: Any, problem: Any, colloc: jax.Array, lr: float,
    steps: int, batch_size: int, x_test: jax.Array, u_exact: jax.Array, *,
    eval_every: int = 100, rad_cfg: Optional[Dict[str, Any]] = None,
):
    params, static = eqx.partition(model, eqx.is_array)
    opt = optax.adam(lr); opt_state = opt.init(params)
    @eqx.filter_jit
    def loss_fn(p, xy): return problem.residual(eqx.combine(p, static), xy)
    @eqx.filter_jit
    def step_fn(p, o, batch):
        loss, g = jax.value_and_grad(loss_fn)(p, batch)
        up, o = opt.update(g, o, p)
        p = eqx.apply_updates(p, up)
        return p, o, loss
    @eqx.filter_jit
    def eval_fn(p):
        m = eqx.combine(p, static)
        pred = jax.vmap(m)(x_test).squeeze()
        return {"rel_l2": _relative_l2(pred, u_exact), "mae": _mae(pred, u_exact),
                "mse": _mse(pred, u_exact), "rmse": _rmse(pred, u_exact)}
    
    loss_hist, metrics_hist, metrics_steps = [], [], []
    best_rel_l2, best_params, best_metrics = np.inf, params, {}
    current_colloc = colloc
    bar = trange(steps, desc="PINN (N=1)", dynamic_ncols=True)
    for s in bar:
        if rad_cfg and s and (s % rad_cfg["resample_every"] == 0):
            key, rad_key = jax.random.split(key)
            sp = rad_cfg["sample_params"].copy()
            sp["n_draw"] = len(current_colloc)
            cur_par, cur_static = eqx.partition(eqx.combine(params, static), eqx.is_array)
            current_colloc = rad_sample(rad_key, problem, cur_par, cur_static, **sp)
        key, sub = jax.random.split(key)
        idx = jax.random.choice(sub, len(current_colloc), (batch_size,), replace=False)
        params, opt_state, loss_val = step_fn(params, opt_state, current_colloc[idx])
        loss_hist.append(float(loss_val))
        if (s + 1) % eval_every == 0 or (s + 1) == steps:
            metrics = {k: float(v) for k, v in eval_fn(params).items()}
            metrics_hist.append(metrics); metrics_steps.append(s + 1)
            if metrics["rel_l2"] < best_rel_l2:
                best_rel_l2, best_params, best_metrics = metrics["rel_l2"], params, metrics
            bar.set_postfix(loss=f"{loss_val:.3e}", L2=f'{metrics["rel_l2"]:.3e}', MAE=f'{metrics["mae"]:.3e}')
        else:
            bar.set_postfix(loss=f"{loss_val:.3e}")
    history = (jnp.array(loss_hist), jnp.array(metrics_steps), metrics_hist)
    final_model = eqx.combine(best_params, static)
    return final_model, best_metrics, history, current_colloc

def train_fbpinn_no_batching(
    key: jax.Array, model: Any, problem: Any, colloc: jax.Array, lr: float,
    steps: int, x_test: jax.Array, u_exact: jax.Array, *,
    eval_every: int = 100, rad_cfg: Optional[Dict[str, Any]] = None,
):
    params, static = eqx.partition(model, eqx.is_array)
    opt = optax.adam(lr); opt_state = opt.init(params)
    @eqx.filter_jit
    def loss_fn(p, xy): return problem.residual(eqx.combine(p, static), xy)
    @eqx.filter_jit
    def step_fn(p, o, full_batch):
        loss, g = jax.value_and_grad(loss_fn)(p, full_batch)
        up, o = opt.update(g, o, p)
        p = eqx.apply_updates(p, up)
        return p, o, loss
    @eqx.filter_jit
    def eval_fn(p):
        m = eqx.combine(p, static)
        pred = jax.vmap(m)(x_test).squeeze()
        return {"rel_l2": _relative_l2(pred, u_exact), "mae": _mae(pred, u_exact),
                "mse": _mse(pred, u_exact), "rmse": _rmse(pred, u_exact)}

    loss_hist, metrics_hist, metrics_steps = [], [], []
    best_rel_l2, best_params, best_metrics = np.inf, params, {}
    current_colloc = colloc
    n_sub = len(model.subnets)
    bar = trange(steps, desc=f"FBPINN (N={n_sub})", dynamic_ncols=True)
    for s in bar:
        if rad_cfg and s and (s % rad_cfg["resample_every"] == 0):
            key, rad_key = jax.random.split(key)
            sp = rad_cfg["sample_params"].copy()
            sp["n_draw"] = len(current_colloc)
            cur_par, cur_static = eqx.partition(eqx.combine(params, static), eqx.is_array)
            current_colloc = rad_sample(rad_key, problem, cur_par, cur_static, **sp)
        params, opt_state, loss_val = step_fn(params, opt_state, current_colloc)
        loss_hist.append(float(loss_val))
        if (s + 1) % eval_every == 0 or (s + 1) == steps:
            metrics = {k: float(v) for k, v in eval_fn(params).items()}
            metrics_hist.append(metrics); metrics_steps.append(s + 1)
            if metrics["rel_l2"] < best_rel_l2:
                best_rel_l2, best_params, best_metrics = metrics["rel_l2"], params, metrics
            bar.set_postfix(loss=f"{loss_val:.3e}", L2=f'{metrics["rel_l2"]:.3e}', MAE=f'{metrics["mae"]:.3e}')
        else:
            bar.set_postfix(loss=f"{loss_val:.3e}")
    history = (jnp.array(loss_hist), jnp.array(metrics_steps), metrics_hist)
    final_model = eqx.combine(best_params, static)
    return final_model, best_metrics, history, current_colloc

# 6. Analysis, Logging and Plotting

def log_final_metrics(save_dir: str, stage_index: int, n_sub: int, model_type: str, metrics: Dict[str, float]):
    os.makedirs(save_dir, exist_ok=True)
    summary_file = os.path.join(save_dir, "final_metrics_summary.csv")
    new_file = not os.path.exists(summary_file)
    with open(summary_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            header = ["timestamp", "stage_index", "n_sub", "model_type", "relative_l2", "mae", "mse", "rmse"]
            writer.writerow(header)
        ts = datetime.now().isoformat()
        row = [ts, stage_index, n_sub, model_type,
               metrics.get('rel_l2', ''), metrics.get('mae', ''),
               metrics.get('mse', ''), metrics.get('rmse', '')]
        writer.writerow(row)

def check_territory(window_fn: Callable, x_test: jax.Array, dominance_threshold: float = 0.8):
    weights = window_fn(x_test)
    num_partitions = int(weights.shape[1])
    info = {"num_partitions": num_partitions, "dominance_threshold": float(dominance_threshold), "missing_indices": [], "weak_peaks": []}
    if num_partitions == 1:
        print(f"       - Health check passed for 1 partition.")
        return True, info
    dominant_partition_indices = jnp.argmax(weights, axis=1)
    unique_dominant_indices = jnp.unique(dominant_partition_indices)
    all_idx = np.arange(num_partitions)
    uniq_np = np.array(unique_dominant_indices)
    missing_indices = np.setdiff1d(all_idx, uniq_np)
    info["missing_indices"] = [int(i) for i in missing_indices.tolist()]
    if len(info["missing_indices"]) > 0:
        print(f"       - HEALTH CHECK FAILED: Partitions {info['missing_indices']} have no territory!")
    for i in range(num_partitions):
        mask = (dominant_partition_indices == i)
        weights_in_territory = weights[mask, i]
        if weights_in_territory.size == 0: continue
        peak = float(jnp.max(weights_in_territory))
        if peak < dominance_threshold: info["weak_peaks"].append((int(i), peak))
    if len(info["weak_peaks"]) > 0:
        peaks_str = ", ".join([f"{i}: {p:.3e}" for i, p in info["weak_peaks"]])
        print(f"       - HEALTH CHECK FAILED: Weak partitions (peak < {dominance_threshold}): {peaks_str}")
    ok = (len(info["missing_indices"]) == 0) and (len(info["weak_peaks"]) == 0)
    if ok: print(f"       - Health check passed for {num_partitions} partitions.")
    return ok, info
    
def log_pou_attempt(save_dir: str, stage_index: int, n_sub: int, status: str, info: Dict[str, Any]):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "pou_attempts_log.csv")
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8") as f:
        if new_file: f.write("timestamp,stage,n_sub,status,missing_indices,weak_peaks,dominance_threshold\n")
        ts = datetime.now().isoformat()
        mi = ";".join(map(str, info.get("missing_indices", [])))
        wp = ";".join([f"{i}:{float(p):.6f}" for i, p in info.get("weak_peaks", [])])
        thr = info.get("dominance_threshold", "")
        f.write(f"{ts},{stage_index},{n_sub},{status},{mi},{wp},{thr}\n")
    if status.lower() == "fail":
        fail_dir = os.path.join(save_dir, "pou_failures")
        os.makedirs(fail_dir, exist_ok=True)
        json_path = os.path.join(fail_dir, f"fail_stage{stage_index}_nsub{n_sub}.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump({"timestamp": datetime.now().isoformat(), "stage": stage_index, "n_sub": n_sub, "status": status, **info}, jf, ensure_ascii=False, indent=2)

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
    if u_pred.shape != u_ex.shape: u_pred, u_ex = u_pred.squeeze(), u_ex.squeeze()
    diff = u_pred - u_ex
    if problem_dim == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(x_test, u_ex, 'k-', label='Exact', linewidth=2)
        ax.plot(x_test, u_pred, 'r--', label='Predicted')
        ax.set_title(f'Stage {stage_index}: FBPINN Result ({n_sub} subdomains)')
        ax.set_xlabel('x'); ax.set_ylabel('u(x)'); ax.legend(); ax.grid(True)
    else:
        u_pred_grid = u_pred.reshape(test_n_2d, test_n_2d); u_ex_grid = u_ex.reshape(test_n_2d, test_n_2d); diff_grid = diff.reshape(test_n_2d, test_n_2d)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        domain_extent = [problem.domain[0][0], problem.domain[1][0], problem.domain[0][1], problem.domain[1][1]]
        vmax, vmin = jnp.max(u_ex_grid), jnp.min(u_ex_grid)
        im1 = axes[0].imshow(u_pred_grid.T, extent=domain_extent, origin='lower', cmap='viridis', vmax=vmax, vmin=vmin)
        axes[0].set_title('Predicted'); fig.colorbar(im1, ax=axes[0])
        im2 = axes[1].imshow(u_ex_grid.T, extent=domain_extent, origin='lower', cmap='viridis', vmax=vmax, vmin=vmin)
        axes[1].set_title('Exact'); fig.colorbar(im2, ax=axes[1])
        vlim = float(jnp.max(jnp.abs(diff_grid)))
        im3 = axes[2].imshow(diff_grid.T, extent=domain_extent, origin='lower', cmap='viridis', vmin=-vlim, vmax=vlim)
        axes[2].set_title('Difference'); fig.colorbar(im3, ax=axes[2])
        fig.suptitle(f'Stage {stage_index}: FBPINN Result ({n_sub} subdomains)', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filepath = os.path.join(save_dir, f"fbpinn_stage_{stage_index}_nsub_{n_sub}.png")
    plt.savefig(filepath, dpi=300); plt.close(fig)
    print(f"           ... plots saved to {filepath}")

def plot_pou_results(window_fn, problem, x_test, stage_index, n_sub_next, save_dir, test_n_2d=100, status_tag: Optional[str] = None):
    status_str = f"({status_tag.upper()})" if status_tag else ""
    print(f"   -> Stage {stage_index}: Generating PoU partition plots for next stage (n_sub={n_sub_next}) {status_str}...")
    problem_dim = len(problem.domain[0]); weights = window_fn(x_test); num_windows = weights.shape[1]
    if problem_dim == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for i in range(weights.shape[1]): ax.plot(x_test, weights[:, i], label=f'Window {i+1}')
        ax.set_title(f'Stage {stage_index}: Learned PoU for {n_sub_next} Subdomains {status_str}'); ax.set_xlabel('x'); ax.set_ylabel('Weight'); ax.legend(); ax.grid(True); ax.set_ylim(-0.1, 1.1)
    else:
        nx, ny = get_grid_dims(num_windows)
        fig, axes = plt.subplots(ny, nx, figsize=(4 * nx, 3.5 * ny), squeeze=False); axes = axes.ravel(); domain_extent = [problem.domain[0][0], problem.domain[1][0], problem.domain[0][1], problem.domain[1][1]]
        for i in range(num_windows):
            ax = axes[i]; window_grid = weights[:, i].reshape(test_n_2d, test_n_2d); im = ax.imshow(window_grid.T, extent=domain_extent, origin='lower', cmap='inferno', vmin=0, vmax=1)
            ax.set_title(f'Window {i+1}'); fig.colorbar(im, ax=ax)
        for j in range(num_windows, len(axes)): axes[j].axis('off')
        fig.suptitle(f'Stage {stage_index}: Learned PoU for {n_sub_next} Subdomains {status_str}', fontsize=16); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    base_filename = f"pou_stage_{stage_index}_nsub_next_{n_sub_next}"
    if status_tag: base_filename += f"_{status_tag}"
    filepath = os.path.join(save_dir, f"{base_filename}.png")
    plt.savefig(filepath, dpi=300); plt.close(fig)
    print(f"           ... plots saved to {filepath}")

def plot_loss_history(loss_hist, metrics_steps, metrics_hist, stage_index, n_sub, save_dir):
    print(f"   -> Stage {stage_index}: Generating loss history plot (n_sub={n_sub})...")
    l2_hist = [m['rel_l2'] for m in metrics_hist]
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(loss_hist, label='Training Loss (MSE)', color='tab:blue', alpha=0.8)
    ax.set_yscale('log'); ax.set_xlabel('Steps'); ax.set_ylabel('Loss', color='tab:blue')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax2 = ax.twinx()
    ax2.plot(metrics_steps, l2_hist, 'o-', label='Evaluation Metric (Relative L2 Error)', color='tab:red')
    ax2.set_ylabel('Relative L2 Error', color='tab:red')
    fig.suptitle(f'Stage {stage_index}: History (n_sub={n_sub})', fontsize=16)
    lines, labels = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filepath = os.path.join(save_dir, f"loss_history_stage_{stage_index}_nsub_{n_sub}.png")
    plt.savefig(filepath, dpi=300); plt.close(fig)
    print(f"           ... plot saved to {filepath}")

def save_npz_history(save_dir, stage_index, n_sub, history_data):
    loss_hist, metrics_steps, metrics_hist = history_data
    filename = f"history_stage_{stage_index}_nsub_{n_sub}.npz"
    filepath = os.path.join(save_dir, filename)
    print(f"   -> Saving training history to {filepath}...")
    l2_hist = [m['rel_l2'] for m in metrics_hist]; mae_hist = [m['mae'] for m in metrics_hist]
    mse_hist = [m['mse'] for m in metrics_hist]; rmse_hist = [m['rmse'] for m in metrics_hist]
    np.savez_compressed(filepath,
                        loss_hist=np.asarray(loss_hist), metrics_steps=np.asarray(metrics_steps),
                        rel_l2_hist=np.asarray(l2_hist), mae_hist=np.asarray(mae_hist),
                        mse_hist=np.asarray(mse_hist), rmse_hist=np.asarray(rmse_hist))
                        
def plot_colloc_points(colloc, domain, stage_id, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.scatter(colloc[:, 0], colloc[:, 1], s=5, alpha=0.6)
    plt.xlim(domain[0][0], domain[1][0]); plt.ylim(domain[0][1], domain[1][1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Collocation Points (Stage: {stage_id}, N={colloc.shape[0]})")
    filepath = os.path.join(save_dir, f"rad_colloc_stage_{stage_id}.png")
    plt.savefig(filepath, dpi=200, bbox_inches='tight'); plt.close()
    print(f"   -> RAD plot saved to {filepath}")

# 7. Hierarchical pipeline

def run_hierarchical_fbpinn(key: jax.Array, problem: Any, config: Dict[str, Any]):
    rad_cfg = config.get("rad_cfg", config.get("rad_config"))
    domain = problem.domain
    test_n_2d, colloc_n_2d = config.get("test_n_2d", 101), config.get("colloc_n_2d", 100)
    x_test = jnp.asarray(generate_collocation(domain, test_n_2d,  strategy="grid"))
    colloc_init = jnp.asarray(generate_collocation(domain, colloc_n_2d, strategy="uniform"))
    u_exact = jax.vmap(problem.exact)(x_test).squeeze()
    save_dir = config["save_dir"]
    if rad_cfg is not None:
        rad_cfg = dict(rad_cfg); rad_cfg.setdefault("sample_params", {})
        rad_cfg["sample_params"]["domain"] = domain
    results: Dict[int, Dict[str, float]] = {}

    key, subkey = jax.random.split(key)
    model = FBPINN_PoU(key=subkey, num_subdomains=1, mlp_config=config["mlp_conf"],
                       domain=domain, ansatz=problem.ansatz, residual_fn=problem.residual)
    model, best_metrics, hist, colloc_cur = train_pinn_with_batching(
        subkey, model, problem, colloc_init, config["FBPINN_LR"], config["FBPINN_STEPS"],
        config["BATCH_SIZE"], x_test, u_exact, rad_cfg=rad_cfg)

    loss_hist, metrics_steps, metrics_hist = hist
    plot_results(model, problem, x_test, u_exact, stage_index=0, n_sub=1, save_dir=save_dir, test_n_2d=test_n_2d)
    plot_loss_history(loss_hist, metrics_steps, metrics_hist, stage_index=0, n_sub=1, save_dir=save_dir)
    save_npz_history(save_dir, stage_index=0, n_sub=1, history_data=hist)
    log_final_metrics(save_dir, stage_index=0, n_sub=1, model_type="Base PINN (Batched)", metrics=best_metrics)
    if rad_cfg and rad_cfg.get("plot_colloc", False):
        plot_colloc_points(colloc_cur, domain, "stage0_final", save_dir)
    results[1] = best_metrics
    current_model = model

    pou_schedule = config.get("pou_schedule", [4, 9, 16, 25])
    discovery_pointer, stage_counter = 0, 1
    problem_dim = len(domain[0])
    pou_type = config.get("pou_type", "sep_mlp")
    dominance_thr = config.get("dominance_threshold", 0.8)

    while discovery_pointer < len(pou_schedule):
        start_idx, last_success_idx, failed_idx = discovery_pointer, None, None
        best_window_fn, best_n_sub = None, None
        print(f"\n[Stage {stage_counter}] PoU discovery starts from schedule index {start_idx}, "
              f"candidates = {pou_schedule[start_idx:]}")
        for i in range(start_idx, len(pou_schedule)):
            n_sub_try = pou_schedule[i]
            try:
                key, pou_key = jax.random.split(key)
                pou_net, init_pars = make_pou_net(
                    pou_type=pou_type, n_sub=n_sub_try, domain=domain,
                    problem_dim=problem_dim, key=pou_key, config=config)
            except ValueError as e:
                print(f"Skip n_sub={n_sub_try}: {e}"); continue
            
            x_pou = jnp.asarray(colloc_cur)
            y_pou = jax.vmap(current_model)(x_pou).squeeze()
            fin_pars = run_lsgd(pou_net, init_pars, x_pou, y_pou, config["lsgd_conf"])
            
            window_fn = partial(pou_net.forward, fin_pars)
            
            ok, info = check_territory(window_fn, x_pou, dominance_threshold=dominance_thr)
            log_pou_attempt(save_dir, stage_counter, n_sub_try, "ok" if ok else "fail", info)
            status_tag = None if ok else "failed"
            plot_pou_results(window_fn, problem, x_test, stage_index=stage_counter,
                             n_sub_next=n_sub_try, save_dir=save_dir, status_tag=status_tag, test_n_2d=test_n_2d)
            if ok:
                last_success_idx, best_n_sub, best_window_fn = i, n_sub_try, window_fn
                print(f"   ✓ Healthy PoU at n_sub={n_sub_try}. Keep pushing...")
            else:
                failed_idx = i
                print(f"   ✗ Health check failed at n_sub={n_sub_try}. Next stage will resume from here."); break
        
        if last_success_idx is None:
            print("No healthy PoU found from current pointer. Stop."); break

        key, subkey = jax.random.split(key)
        model = FBPINN_PoU(key=subkey, num_subdomains=best_n_sub, mlp_config=config["mlp_conf"],
                           window_fn=best_window_fn, domain=domain, ansatz=problem.ansatz,
                           residual_fn=problem.residual)
        model, best_metrics, hist, colloc_cur = train_fbpinn_no_batching(
            subkey, model, problem, colloc_cur, config["FBPINN_LR"], config["FBPINN_STEPS"],
            x_test, u_exact, rad_cfg=rad_cfg)
        
        loss_hist, metrics_steps, metrics_hist = hist
        plot_results(model, problem, x_test, u_exact, stage_counter, best_n_sub, save_dir, test_n_2d=test_n_2d)
        plot_loss_history(loss_hist, metrics_steps, metrics_hist, stage_counter, best_n_sub, save_dir)
        save_npz_history(save_dir, stage_index=stage_counter, n_sub=best_n_sub, history_data=hist)
        log_final_metrics(save_dir, stage_index=stage_counter, n_sub=best_n_sub, model_type="FBPINN (Full-batch)", metrics=best_metrics)
        if rad_cfg and rad_cfg.get("plot_colloc", False):
            plot_colloc_points(colloc_cur, domain, f"stage{stage_counter}_final", save_dir)
        results[best_n_sub] = best_metrics
        current_model = model
        stage_counter += 1
        discovery_pointer = failed_idx if failed_idx is not None else last_success_idx + 1
        
    return results


if __name__ == '__main__':
    config = {
        "BATCH_SIZE": 10000, "FBPINN_STEPS": 30, "FBPINN_LR": 1e-3,
        "test_n_2d": 101, "colloc_n_2d": 101,
        "mlp_conf": dict(in_size=2, out_size=1, width_size=64, depth=2, activation=jnp.tanh),
        "lsgd_conf": LSGDConfig(n_epochs=100, lr=1e-3),
        "pou_type": "sep_mlp", "dominance_threshold": 0.8,
        
        "sep_rbf_pou_conf": dict(tau=1.0),
        "sep_mlp_pou_conf": dict(hidden=(32, 32), tau=1.0),
        "mlp_pou_conf": dict(hidden=(16, 16)),
        "pou_schedule": [4, 9, 16, 25, 36, 49, 64, 81, 100],

        "rad_cfg": {
            "resample_every": 10000,
            "plot_colloc": True,
            "sample_params": {
                "pool_size": 20000,
                "k": 1.0,
                "c": 1.0,
            }
        }
    }
    
    problem = FirstOrderFreq1010()
    key = jax.random.PRNGKey(42)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config["save_dir"] = f"results_{problem.__class__.__name__}_{timestamp}_RAD_ON"
    os.makedirs(config["save_dir"], exist_ok=True)
    print(f"Results will be saved to: {config['save_dir']}")

    final_results = run_hierarchical_fbpinn(key, problem, config)

    print("\n\n" + "#"*100)
    print("##### Algorithm execution finished #####")
    print("#"*100)
    print(f"All results, plots, and history have been saved to: '{config['save_dir']}'")
    print(f"A summary of final metrics for all stages has been saved to: '{os.path.join(config['save_dir'], 'final_metrics_summary.csv')}'")
    print("\nSummary of Final Metrics for each stage:")
    header = f"   {'Model':<25} | {'n_sub':<5} | {'Relative L2':<15} | {'MAE':<15} | {'MSE':<15} | {'RMSE':<15}"
    print(header)
    print("   " + "-" * (len(header) - 3))
    sorted_results = sorted(final_results.items())
    for n_sub_val, metrics in sorted_results:
        model_type = "Base PINN " if n_sub_val == 1 else "FBPINN"
        l2_str = f"{metrics.get('rel_l2', 0):.4e}"
        mae_str = f"{metrics.get('mae', 0):.4e}"
        mse_str = f"{metrics.get('mse', 0):.4e}"
        rmse_str = f"{metrics.get('rmse', 0):.4e}"
        print(f"   {model_type:<25} | {n_sub_val:<5} | {l2_str:<15} | {mae_str:<15} | {mse_str:<15} | {rmse_str:<15}")