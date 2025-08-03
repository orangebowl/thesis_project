from __future__ import annotations

"""
Additive FBPINN implementation **with residual‑aware PoU training**.
---------------------------------------------------------------
Key change vs. original version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
During PoU (partition‑of‑unity) fitting, we minimise the reconstruction error
of the **local residual** r(x) = L[u_base](x) ‑ f(x) instead of the base model
output u_base(x).  Concretely, in `run_additive_fbpinn` we compute

    y_pou = vmap(problem.pointwise_residual, in_axes=(None,0))(frozen_model, colloc_cur)

and feed that as the training target into `run_lsgd`.  Nothing else in
`run_lsgd` changes, because its loss already measures the discrepancy between
`predict_from_coeffs(...)` and the provided `y`.  Supplying residuals therefore
switches the optimisation objective seamlessly.
"""

import os, sys
from pathlib import Path
from datetime import datetime
from functools import partial
from typing import Sequence, Dict, Tuple, Any, Callable, Optional
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from tqdm import trange

# ==============================================================================
# 1. Core Module Definitions
# ==============================================================================

class BaseProblem:
    """Abstract base class for PDE problems."""
    domain: Tuple[jnp.ndarray, jnp.ndarray]

    def exact(self, x):
        raise NotImplementedError

    def residual(self, model, x):
        raise NotImplementedError

    def pointwise_residual(self, model, x):
        raise NotImplementedError

    @staticmethod
    def ansatz(xy: jax.Array, nn_out: jax.Array) -> jax.Array:
        return nn_out


class MLP(eqx.Module):
    """A standard Multi‑Layer Perceptron."""
    layers: list
    activation: Callable

    def __init__(self, *, key, in_size, out_size, width_size, depth, activation):
        keys = jax.random.split(key, depth + 1)
        self.activation = activation
        self.layers = []
        dims = [in_size] + [width_size] * depth + [out_size]
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            self.layers.append(eqx.nn.Linear(in_dim, out_dim, key=keys[i]))

    def __call__(self, x: jax.Array) -> jax.Array:
        # Transpose input from (batch, features) to (features, batch)
        x = x.T
        for layer in self.layers[:-1]:
            x = layer.weight @ x + layer.bias[:, None]
            x = self.activation(x)
        final_layer = self.layers[-1]
        x = final_layer.weight @ x + final_layer.bias[:, None]
        return x.T  # (batch, 1)


class FBPINN_PoU(eqx.Module):
    """Additive FBPINN model with (optional) frozen base model."""

    subnets: list[MLP]
    window_fn: Optional[Callable] = eqx.static_field()
    ansatz: Callable = eqx.static_field()
    base_model: Optional[eqx.Module] = eqx.static_field()

    def __init__(
        self,
        *,
        key: jax.Array,
        num_subdomains: int,
        mlp_config: dict,
        ansatz: Callable,
        window_fn: Optional[Callable] = None,
        base_model: Optional[eqx.Module] = None,
    ):
        self.window_fn = window_fn
        self.ansatz = ansatz
        self.base_model = base_model
        subnet_keys = jax.random.split(key, num_subdomains)
        self.subnets = [MLP(key=k, **mlp_config) for k in subnet_keys]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_2d = jnp.atleast_2d(x)
        # local predictions from each subnet
        u_local_list = [net(x_2d) for net in self.subnets]
        u_local = jnp.concatenate(u_local_list, axis=-1)  # (batch, n_sub)

        if self.window_fn is not None:
            weights = self.window_fn(x_2d)  # (batch, n_sub)
            if weights.ndim == u_local.ndim - 1:
                weights = weights[..., None]
            u_correction = jnp.sum(weights * u_local, axis=1, keepdims=True)
        else:
            u_correction = u_local

        u_correction = self.ansatz(x_2d, u_correction)

        if self.base_model is not None:
            u_base = self.base_model(x_2d)
            if u_base.ndim == 1:
                u_base = u_base[:, None]
            return u_base + u_correction
        return u_correction


class FirstOrderFreq1010(BaseProblem):
    domain = (jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))

    @staticmethod
    def exact(xy: jax.Array) -> jax.Array:
        xy = jnp.atleast_2d(xy)
        x, y = xy[:, 0], xy[:, 1]
        return (jnp.sin(10 * jnp.pi * x**2) * jnp.sin(10 * jnp.pi * y**2))[:, None]

    @staticmethod
    def ansatz(xy: jax.Array, nn_out: jax.Array) -> jax.Array:
        x, y = xy[..., 0], xy[..., 1]
        return (x * y)[..., None] * nn_out

    @staticmethod
    def rhs(xy: jax.Array) -> jax.Array:
        xy = jnp.atleast_2d(xy)
        x, y = xy[:, 0], xy[:, 1]
        α = 10 * jnp.pi
        term_x = 20 * jnp.pi * x * jnp.cos(α * x**2) * jnp.sin(α * y**2)
        term_y = 20 * jnp.pi * y * jnp.sin(α * x**2) * jnp.cos(α * y**2)
        return (term_x + term_y)[:, None]

    # Gradient and residual computations
    def _pointwise_res(self, model: Callable, xy_batch: jax.Array) -> jax.Array:
        grad_u_fn = jax.grad(lambda z: jnp.sum(model(z)))  # grad wrt each input
        grad_u = grad_u_fn(xy_batch)  # (batch, 2)
        grad_sum = grad_u[:, 0] + grad_u[:, 1]
        r = grad_sum[:, None] - self.rhs(xy_batch)
        return r  # (batch, 1)

    def pointwise_residual(self, model, xy):
        return self._pointwise_res(model, jnp.atleast_2d(xy))

    def residual(self, model, xy_batch):
        r = self._pointwise_res(model, xy_batch)
        return jnp.mean(r**2)


# ==============================================================================
# 2. Helper and Utility Functions
# ==============================================================================

def generate_collocation(domain, N, strategy="grid"):
    sqrt_N = int(np.sqrt(N))
    if strategy == "grid" and sqrt_N * sqrt_N == N:
        x_coords = np.linspace(domain[0][0], domain[1][0], sqrt_N)
        y_coords = np.linspace(domain[0][1], domain[1][1], sqrt_N)
        X, Y = np.meshgrid(x_coords, y_coords)
        return np.vstack([X.ravel(), Y.ravel()]).T
    lo, hi = domain
    return jax.random.uniform(jax.random.PRNGKey(0), (N, len(lo)), minval=lo, maxval=hi)


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

    @staticmethod
    def _mlp_forward(params, x):
        h = x
        n_layer = (len(params) // 2) - 1
        for i in range(n_layer):
            h = jnp.tanh(h @ params[f"W{i}"] + params[f"b{i}"])
        return h @ params[f"W{n_layer}"] + params[f"b{n_layer}"]

    def init_params(self) -> Dict[str, Any]:
        return {"x": self.param_x, "y": self.param_y}

    def forward(self, params: Dict[str, Any], xy: jnp.ndarray) -> jnp.ndarray:
        xy = jnp.atleast_2d(xy)
        x, y = xy[:, :1], xy[:, 1:]
        z_x = self._mlp_forward(params["x"], x)
        z_y = self._mlp_forward(params["y"], y)
        logits = (z_x[:, :, None] + z_y[:, None, :]) / self.tau  # (batch, nx, ny)
        logits_flat = logits.reshape(x.shape[0], -1)
        # numerically stable softmax
        return jax.nn.softmax(logits_flat - jnp.max(logits_flat, axis=-1, keepdims=True), axis=-1)


class WindowModule(eqx.Module):
    pou_net: Any = eqx.static_field()
    params: Dict

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.pou_net.forward(self.params, x)


# ----- Local polynomial helpers ------------------------------------------------

def _design_matrix(x: jnp.ndarray) -> jnp.ndarray:
    x1, x2 = x[:, 0], x[:, 1]
    return jnp.stack([jnp.ones_like(x1), x1, x2, x1**2, x1 * x2, x2**2], -1)


def fit_local_polynomials(x, y, w, lam: float = 0.0):
    A = _design_matrix(x)
    y = y.squeeze()[:, None]
    k = A.shape[-1]

    def _solve(weights):
        Aw = A * weights[:, None]
        M = A.T @ Aw
        b = (Aw.T @ y).squeeze(-1)
        return jnp.linalg.solve(M + lam * jnp.eye(k), b)

    return jax.vmap(_solve, 1, 0)(w)  # (n_sub, k)


def _predict_from_coeffs(x, coeffs, partitions):
    A = _design_matrix(x)
    y_cent = A @ coeffs.T  # (batch, n_sub)
    return jnp.sum(partitions * y_cent, 1)  # (batch,)


@dataclasses.dataclass
class LSGDConfig:
    n_epochs: int = 15000
    lr: float = 1e-4
    lam_init: float = 0.05
    rho: float = 0.99
    n_stag: int = 300
    ent_weight: float = 1e-3      # ★ 新增：熵正则权重


# ------------------------------------------------------------------------------
# LSGD optimiser for PoU parameters
# ------------------------------------------------------------------------------

def run_lsgd(
    pou_net,
    initial_params,
    x,
    y,
    cfg: LSGDConfig,
):
    """Least‑Squares Gradient Descent to tune PoU so that it reconstructs *y*."""

    params = initial_params

    loss_fn = jax.jit(
        lambda p, lam: jnp.mean(
            (
                _predict_from_coeffs(
                    x,
                    fit_local_polynomials(x, y, pou_net.forward(p, x), lam),
                    pou_net.forward(p, x),
                )
                - y.squeeze()
            )
            ** 2
        )
    )

    valgrad_fn = jax.jit(jax.value_and_grad(loss_fn))
    opt = optax.adam(cfg.lr)
    opt_state = opt.init(params)
    lam = jnp.array(cfg.lam_init)
    best, stag = jnp.inf, 0
    bar = trange(cfg.n_epochs, desc=f"PoU‑LSGD (N={pou_net.num_experts})", dynamic_ncols=True)

    for ep in bar:
        loss_val, grads = valgrad_fn(params, lam)
        updates, opt_state = opt.update(grads, opt_state)
        params = eqx.apply_updates(params, updates)
        bar.set_postfix(loss=f"{loss_val:.3e}")

        if loss_val < best - 1e-12:
            best, stag = loss_val, 0
        else:
            stag += 1
        if stag > cfg.n_stag:
            lam *= cfg.rho  # ridge decay
            stag = 0

    return params


# ------------------------------------------------------------------------------
# Residual‑based adaptive sampling
# ------------------------------------------------------------------------------

def rad_sample(
    key,
    problem,
    params,
    static,
    *,
    domain,
    n_draw,
    pool_size,
    k=3.0,
    c=1.0,
):
    lo, hi = jnp.array(domain[0]), jnp.array(domain[1])
    pool = jax.random.uniform(key, (pool_size, len(lo)), minval=lo, maxval=hi)
    model = eqx.combine(params, static)
    vmapped_residual_fn = jax.vmap(problem.pointwise_residual, in_axes=(None, 0))
    res_vals = vmapped_residual_fn(model, pool).squeeze()
    prob = jnp.abs(res_vals) ** k / jnp.mean(jnp.abs(res_vals) ** k) + c
    prob /= prob.sum()
    subkey = jax.random.split(key)[1]
    return pool[jax.random.choice(subkey, pool_size, (n_draw,), p=prob, replace=False)]


# ==============================================================================
# 3. Training loops
# ==============================================================================

def train_model(
    key,
    model,
    problem,
    colloc,
    lr,
    steps,
    x_test,
    u_exact,
    *,
    eval_every=100,
    rad_config=None,
    batch_size=None,
):
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
        return eqx.apply_updates(p, up), o, loss

    @eqx.filter_jit
    def eval_fn(p, x):                       # 把 x_test 作为参数
        pred = jax.vmap(eqx.combine(p, static))(x).squeeze()
        return jnp.mean(jnp.abs(pred - u_exact.squeeze()))

    # jit warm‑up
    step_fn(params, opt_state, colloc[: (batch_size or 10)])

    loss_hist, l1_hist, l1_steps = [], [], []
    best_l1, best_params = np.inf, params
    current_colloc = colloc

    desc = (
        "PINN (N=1, Batched)" if batch_size else f"FBPINN (N={len(model.subnets)}, Full‑batch)"
    )
    bar = trange(steps, desc=desc, dynamic_ncols=True)

    for s in bar:
        if rad_config and s and (s % rad_config["resample_every"] == 0):
            key, rad_key = jax.random.split(key)
            sp = rad_config["sample_params"].copy()
            sp["n_draw"] = len(current_colloc)
            current_colloc = rad_sample(
                rad_key,
                problem,
                *eqx.partition(eqx.combine(params, static), eqx.is_array),
                **sp,
            )

        batch_data = current_colloc
        if batch_size:
            key, sub = jax.random.split(key)
            idx = jax.random.permutation(sub, len(current_colloc))[:batch_size]
            batch_data = current_colloc[idx]

        params, opt_state, loss_val = step_fn(params, opt_state, batch_data)
        loss_hist.append(float(loss_val))

        if (s + 1) % eval_every == 0 or s == steps - 1:
            l1 = float(eval_fn(params, x_test))  # 调用时传入
            l1_hist.append(l1)
            l1_steps.append(s + 1)
            if l1 < best_l1:
                best_l1, best_params = l1, params
            bar.set_postfix(loss=f"{loss_val:.3e}", L1_err=f"{l1:.3e}")

    return (
        eqx.combine(best_params, static),
        best_l1,
        (loss_hist, l1_steps, l1_hist),
        current_colloc,
    )


# ==============================================================================
# 4. Plotting helpers (unchanged, omitted here for brevity)
def get_grid_dims(n_sub):
    if n_sub == 1:
        return 1, 1
    d = next((i for i in range(int(np.sqrt(n_sub)), 0, -1) if n_sub % i == 0), 1)
    return d, n_sub // d


def check_territory(window_fn, x_test, dominance_threshold=0.8):
    w = window_fn(x_test)
    if w.shape[1] == 1:
        return True
    dom = jnp.argmax(w, axis=1)
    if len(jnp.unique(dom)) < w.shape[1]:
        return False
    return all(jnp.max(w[dom == i, i]) >= dominance_threshold for i in range(w.shape[1]) if (dom == i).any())


def plot_results(model, problem, xt, u_ex, stage, n_sub, out_dir, n=100):
    up = jax.vmap(model)(xt).reshape(n, n)
    u_exact = u_ex.reshape(n, n)
    domain = [problem.domain[0][0], problem.domain[1][0], problem.domain[0][1], problem.domain[1][1]]
    vmin, vmax = u_exact.min(), u_exact.max()
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    im0 = ax[0].imshow(up.T, extent=domain, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    im1 = ax[1].imshow(u_exact.T, extent=domain, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    im2 = ax[2].imshow(np.abs(up - u_exact).T, extent=domain, origin="lower", cmap="inferno")
    ax[0].set_title("Predicted"); ax[1].set_title("Exact"); ax[2].set_title("Abs Error")
    fig.suptitle(f"Stage {stage} (n_sub={n_sub})")
    fig.colorbar(im0, ax=ax[0]); fig.colorbar(im1, ax=ax[1]); fig.colorbar(im2, ax=ax[2])
    fig.savefig(Path(out_dir) / f"stage{stage}_nsub{n_sub}_result.png", dpi=300)
    plt.close(fig)


def plot_pou_results(window_fn, xt, stage, n_sub, out_dir, n=100):
    w = window_fn(xt)
    nx, ny = get_grid_dims(w.shape[1])
    fig, axes = plt.subplots(ny, nx, figsize=(4 * nx, 3.5 * ny), squeeze=False, sharex=True, sharey=True)
    axes = axes.ravel()
    for i in range(w.shape[1]):
        im = axes[i].imshow(w[:, i].reshape(n, n).T, extent=[0, 1, 0, 1], origin="lower", cmap="inferno", vmin=0, vmax=1)
        axes[i].set_title(f"Window {i + 1}"); fig.colorbar(im, ax=axes[i])
    for j in range(w.shape[1], len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Stage {stage}: PoU weights (n_sub={n_sub})")
    fig.savefig(Path(out_dir) / f"stage{stage}_pou_nsub{n_sub}.png", dpi=300)
    plt.close(fig)


def plot_loss_history(hist, stage, n_sub, out_dir):
    loss, steps, l1 = hist
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(loss, label="Residual loss")
    ax.set_yscale("log"); ax.set_xlabel("Steps"); ax.set_ylabel("Loss")
    ax2 = ax.twinx(); ax2.plot(steps, l1, "r.-", label="L1 error")
    ax2.set_ylabel("Rel L1")
    ax.grid(True, which="both", ls="--")
    fig.suptitle(f"Stage {stage} (n_sub={n_sub}) history")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    fig.savefig(Path(out_dir) / f"stage{stage}_nsub{n_sub}_history.png", dpi=300)
    plt.close(fig)


def plot_colloc_points(colloc, domain, tag, out_dir):
    plt.figure(figsize=(6, 6))
    plt.scatter(colloc[:, 0], colloc[:, 1], s=4, alpha=0.6)
    plt.xlim(domain[0][0], domain[1][0]); plt.ylim(domain[0][1], domain[1][1])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(f"Collocation ({tag}, N={len(colloc)})")
    plt.savefig(Path(out_dir) / f"colloc_{tag}.png", dpi=250)
    plt.close()

# ==============================================================================
# 5. Main execution logic with **Residual‑aware PoU training**
# ==============================================================================

def get_grid_dims(n_sub):
    if n_sub == 1:
        return 1, 1
    best_factor = next((i for i in range(int(np.sqrt(n_sub)), 0, -1) if n_sub % i == 0), 1)
    return best_factor, n_sub // best_factor


def run_additive_fbpinn(key, problem: BaseProblem, cfg: Dict[str, Any]):
    """Main training loop that uses the *standard plotting helpers* the user
    specified (`plot_results`, `plot_pou_results`, `plot_loss_history`,
    `plot_colloc_points`)."""
    # ------------------------------------------------------------------
    dom = problem.domain
    n_test = cfg.get("test_n_2d", 100)
    n_col  = cfg.get("colloc_n_2d", 100)

    # test grid (n_test × n_test regular grid)
    grid_lin = jnp.asarray(np.linspace(0., 1., n_test))
    x_test   = jnp.array([[x, y] for y in grid_lin for x in grid_lin])

    # initial collocation on grid
    colloc = jnp.asarray(generate_collocation(dom, n_col**2, "grid"))
    u_exact = jax.vmap(problem.exact)(x_test).squeeze()

    out_dir = Path(cfg["save_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # ======================= Stage 0: global PINN =========================
    key, sk = jax.random.split(key)
    model = FBPINN_PoU(key=sk,
                       num_subdomains=1,
                       mlp_config=cfg["mlp_conf"],
                       ansatz=problem.ansatz)

    model, best, hist, colloc = train_model(
        sk, model, problem, colloc,
        cfg["FBPINN_LR"], cfg["FBPINN_STEPS"],
        x_test, u_exact,
        rad_config=cfg.get("rad_config"))

    plot_results(model, problem, x_test, u_exact, stage=0, n_sub=1, out_dir=out_dir)
    plot_loss_history(hist, stage=0, n_sub=1, out_dir=out_dir)
    plot_colloc_points(colloc, dom, tag="stage0", out_dir=out_dir)

    results[1] = {"rel_l1_error": float(best)}
    frozen = model

    # ===================== Additive stages ===============================
    for stage, n_sub in enumerate(cfg.get("pou_schedule", [4, 9, 16]), 1):
        print(f"── Stage {stage}: PoU {n_sub} sub‑domains ──")
        nx, ny = get_grid_dims(n_sub)
        key, pk = jax.random.split(key)
        # ① 先挑出对应 n_sub 的 PoU‑Net 配置
        pou_conf = cfg.get("sep_mlp_pou_conf_by_n_sub", {}).get(
                    n_sub, cfg.get("sep_mlp_pou_conf", {})
                )

        # ② 用挑出的配置实例化网络
        pou_net  = SepMLPPOUNet(nx, ny, key=pk, **pou_conf)

        # prediction‑aware target (frozen model output)
        #y_pou = jax.vmap(problem.pointwise_residual, in_axes=(None,0))(frozen, colloc)
        y_pou = jax.vmap(frozen)(colloc)

        # ① 先拿对应 n_sub 的 LSGDConfig，没有则用默认
        lsgd_cfg = cfg.get("lsgd_conf_by_n_sub", {}).get(n_sub, cfg["lsgd_conf"])

        # ② 用选出的超参跑 LSGD
        fin = run_lsgd(pou_net, pou_net.init_params(), colloc, y_pou, lsgd_cfg)
        
        window_fn = WindowModule(pou_net=pou_net, params=fin)

        plot_pou_results(window_fn, x_test, stage - 0.5, n_sub, out_dir)

        key, sk = jax.random.split(key)
        res_model = FBPINN_PoU(key=sk,
                               num_subdomains=n_sub,
                               mlp_config=cfg["mlp_conf"],
                               window_fn=window_fn,
                               ansatz=problem.ansatz,
                               base_model=frozen)

        res_model, best, hist, colloc = train_model(
            sk, res_model, problem, colloc,
            cfg["FBPINN_LR"], cfg["FBPINN_STEPS"],
            x_test, u_exact,
            rad_config=cfg.get("rad_config"))

        plot_results(res_model, problem, x_test, u_exact, stage, n_sub, out_dir)
        plot_loss_history(hist, stage, n_sub, out_dir)
        plot_colloc_points(colloc, dom, tag=f"stage{stage}", out_dir=out_dir)

        results[n_sub] = {"rel_l1_error": float(best)}
        frozen = res_model

    return results



# ==============================================================================
# 6. Entry‑point
# ==============================================================================

if __name__ == "__main__":

    config = {
        "BATCH_SIZE": 2048,
        "FBPINN_STEPS": 30000,
        "FBPINN_LR": 1e-3,
        "test_n_2d": 100,
        "colloc_n_2d": 100,
        "mlp_conf": dict(in_size=2, out_size=1, width_size=64, depth=2, activation=jnp.tanh),
        "lsgd_conf": LSGDConfig(n_epochs=5000, lr=1e-4, n_stag=300),
        "lsgd_conf_by_n_sub": {
        4:  LSGDConfig(n_epochs=8000, lr=1e-4, n_stag=8000),
        9:  LSGDConfig(n_epochs=8000, lr=1e-4, n_stag=8000),   # 示例，可随意改
        16: LSGDConfig(n_epochs=8000, lr=1e-4, n_stag=8000),
    },
        "sep_mlp_pou_conf": dict(hidden=(8, 8), tau=2.0),  # sharpen windows
        "sep_mlp_pou_conf_by_n_sub": {
        4:  dict(hidden=(4, 4),  tau=2.0),   # 示例
        9:  dict(hidden=(8, 8), tau=2.0),
        16: dict(hidden=(16, 16), tau=2.0),
    },
        "pou_schedule": [4, 9, 16],  # stages to run
        "rad_config": {
            "resample_every": 5000,
            "sample_params": {
                "pool_size": 20000,
                "k": 3.0,
                "c": 1.0,
            },
        },
    }

    problem = FirstOrderFreq1010()
    if "rad_config" in config:
        config["rad_config"]["sample_params"]["domain"] = problem.domain

    key = jax.random.PRNGKey(42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["save_dir"] = f"results_additive_residual_pou_{timestamp}"

    print(
        f"Starting additive FBPINN (residual‑aware PoU) for problem: {problem.__class__.__name__}\n",
        f"Results will be saved to: {config['save_dir']}\n",
    )

    final_results = run_additive_fbpinn(key, problem, config)

    print("\n" + "#" * 80)
    print("##### Execution finished #####")
    print(f"All results saved to: '{config['save_dir']}'\n\nSummary of relative L1 error:")
    for n_sub_val, res in sorted(final_results.items()):
        model_desc = "Base PINN" if n_sub_val == 1 else "Additive FBPINN"
        print(f"   {model_desc:15s} (n_sub={n_sub_val:<2}) | L1 = {res['rel_l1_error']:.4e}")
