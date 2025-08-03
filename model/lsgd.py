import jax
import jax.numpy as jnp
import optax
import dataclasses
from model.POU_nets import BasePOUNet
from vis.vis_pou import viz_partitions


# --------- 多项式设计矩阵 & 局部拟合 ----------------------------------
def _design_matrix(x: jnp.ndarray) -> jnp.ndarray:
    """支持 1D/2D：返回二次多项式 Vandermonde"""
    d = x.shape[-1]
    if d == 1:                   # [1, x, x^2]
        x1 = x[:, 0]
        return jnp.stack([jnp.ones_like(x1), x1, x1**2], -1)
    elif d == 2:                 # [1, x, y, x^2, xy, y^2]
        x1, x2 = x[:, 0], x[:, 1]
        return jnp.stack([jnp.ones_like(x1), x1, x2, x1**2, x1*x2, x2**2], -1)
    else:
        raise ValueError("Only 1-D or 2-D supported")

def _poly_dim(input_dim: int) -> int:
    return 3 if input_dim == 1 else 6       # 二次多项式参数个数

def fit_local_polynomials(x, y, w, lam: float = 0.0):
    """batch 求解每个 partition 的加权二次多项式系数"""
    A, y = _design_matrix(x), y[:, None]
    k = A.shape[-1]                         # 3 或 6

    def _solve(weights):
        Aw = A * weights[:, None]
        M  = A.T @ Aw
        b  = (Aw.T @ y).squeeze(-1)
        return jnp.linalg.solve(M + lam*jnp.eye(k), b)

    return jax.vmap(_solve, 1, 0)(w)        # (C,k)

def _predict_from_coeffs(x, coeffs, partitions):
    A = _design_matrix(x)                   # (N,k)
    y_cent = A @ coeffs.T                  # (N,C)
    return jnp.sum(partitions * y_cent, 1) # (N,)


@dataclasses.dataclass
class LSGDConfig:
    n_epochs: int   = 5000
    lr: float       = 1e-3
    lam_init: float = 5e-4
    rho: float      = 0.99
    n_stag: int     = 100
    prints: int     = 10
    viz_int: int = 200   # None = no plot


def run_lsgd(model: BasePOUNet, params: dict, x, y, cfg: LSGDConfig):
    lam = jnp.array(cfg.lam_init)
    best, stag = jnp.inf, 0
    log_int = max(1, cfg.n_epochs//cfg.prints)

    @jax.jit
    def loss_fn(p, lam_):
        part   = model.forward(p, x)
        coeffs = fit_local_polynomials(x, y, part, lam_)
        pred   = _predict_from_coeffs(x, coeffs, part)
        return jnp.mean((pred - y)**2)

    valgrad = jax.jit(lambda p, l: jax.value_and_grad(
        lambda pp: loss_fn(pp,l))(p))

    opt = optax.adam(cfg.lr); opt_state = opt.init(params)

    print(" JIT compiling ...", end="", flush=True)
    loss_val, grads = valgrad(params, lam); print(" done")

    for ep in range(cfg.n_epochs):
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        loss_val, grads = valgrad(params, lam)

        if cfg.viz_int and ep % cfg.viz_int == 0:
            viz_partitions(model, params, title=f"epoch {ep}")

        if ep % log_int == 0:
            print(f"epoch {ep:6d} | loss {loss_val:.6e} | λ={float(lam):.1e}")

        # λ decay
        if loss_val < best - 1e-12:
            best, stag = loss_val, 0
        else:
            stag += 1
        if stag > cfg.n_stag:
            lam *= cfg.rho; stag = 0

    return params
