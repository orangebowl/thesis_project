import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from tqdm import trange
from typing import Callable, Optional, Dict, Any, Tuple

from pathlib import Path
import matplotlib.pyplot as plt


def _save_colloc_snapshot(colloc: jax.Array, domain: Tuple[jax.Array, jax.Array], out_dir: str, tag: str) -> None:
    """Save a snapshot of collocation points after a RAD resample (supports 1D/2D)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))

    if colloc.shape[1] == 1:
        x = colloc.squeeze()
        plt.scatter(x, jnp.zeros_like(x), s=3, alpha=0.5)
        plt.yticks([])
        plt.xlabel("x")
        plt.xlim(float(domain[0][0]), float(domain[1][0]))
        plt.title(f"RAD collocation after {tag} (N={colloc.shape[0]})")
    elif colloc.shape[1] == 2:
        plt.scatter(colloc[:, 0], colloc[:, 1], s=3, alpha=0.5)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(float(domain[0][0]), float(domain[1][0]))
        plt.ylim(float(domain[0][1]), float(domain[1][1]))
        plt.title(f"RAD collocation after {tag} (N={colloc.shape[0]})")
    else:
        plt.close()
        return  # skip 3D+

    out_path = Path(out_dir) / f"rad_{tag}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


@eqx.filter_jit
def pointwise_residual(problem, params, static, x: jax.Array) -> jax.Array:
    """Return |r(x)| per sample as shape (N,). Uses a per-point vmap."""
    model = eqx.combine(params, static)

    def _single_point_residual(xi):
        # residual expects a batch; add batch dim and take sqrt to get |r|
        return jnp.sqrt(problem.residual(model, xi[None, ...]))

    return jax.vmap(_single_point_residual)(x).reshape(-1)


def rad_sample(
    key: jax.random.PRNGKey,
    problem,
    params,
    static,
    *,
    domain: Tuple[jax.Array, jax.Array],
    n_draw: int,
    pool_size: int,
    k: float = 1.0,
    c: float = 1.0,
) -> jax.Array:
    """Residual-based Adaptive Sampling (RAD)."""
    lo, hi = jnp.asarray(domain[0]), jnp.asarray(domain[1])
    xdim = lo.size

    # candidate pool
    pool = jax.random.uniform(key, (pool_size, xdim), minval=lo, maxval=hi)

    # weights from residuals
    key, sk = jax.random.split(key)
    r = pointwise_residual(problem, params, static, pool)  # (pool_size,)
    w = (r**k) / (jnp.mean(r**k) + 1e-9) + c               # positive + stabilized
    prob = w / jnp.sum(w)

    choices = jnp.arange(pool_size)

    # sample without replacement
    if n_draw >= pool_size:
        idx = jax.random.permutation(sk, choices)[:n_draw]
    else:
        idx = jax.random.choice(sk, choices, shape=(n_draw,), p=prob, replace=False)

    return pool[idx]


def train_fbpinn(
    *,
    key: jax.random.PRNGKey,
    model: eqx.Module,
    problem,
    colloc: jax.Array,
    lr: float = 1e-3,
    steps: int = 10000,
    x_test: Optional[jax.Array] = None,
    u_exact: Optional[Callable[[jax.Array], jax.Array]] = None,
    rad_cfg: Optional[Dict[str, Any]] = None,
    eval_every: int = 100,
) -> Tuple[eqx.Module, jax.Array, Tuple[jax.Array, jax.Array]]:
    """Train an FBPINN model, optionally with RAD resampling."""
    colloc = colloc.astype(jnp.float32)
    params, static = eqx.partition(model, eqx.is_array)

    build_model = lambda p: eqx.combine(p, static)
    loss_fn = lambda p, xy: problem.residual(build_model(p), xy)

    @eqx.filter_jit
    def eval_fn_rel_l2(p):
        if x_test is None or u_exact is None:
            return jnp.nan
        pred = build_model(p)(x_test.astype(jnp.float32)).squeeze()
        exact = u_exact(x_test).astype(jnp.float32).squeeze()
        num = jnp.linalg.norm(pred - exact)
        denom = jnp.linalg.norm(exact)
        return num / (denom + 1e-12)

    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @eqx.filter_jit
    def step_fn(p, o, xy):
        xy = xy.reshape(-1, xy.shape[-1])
        loss, grads = jax.value_and_grad(loss_fn)(p, xy)
        updates, o = opt.update(grads, o, p)
        p = eqx.apply_updates(p, updates)
        return p, o, loss

    # ahead-of-time compile
    _ = step_fn(params, opt_state, colloc[:1])

    use_rad = rad_cfg is not None
    if use_rad:
        resample_every = int(rad_cfg["resample_every"])
        pool_size = int(rad_cfg["pool_size"])
        k = float(rad_cfg["k"])
        c = float(rad_cfg["c"])
        viz_dir = rad_cfg.get("viz_dir", None)
        viz_every = int(rad_cfg.get("viz_every", 1))
        print(f"[RAD] Resampling every {resample_every} steps (pool={pool_size}, k={k}, c={c})")

    loss_hist, rel_l2_hist, rel_l2_steps = [], [], []
    bar = trange(steps, dynamic_ncols=True, desc="FBPINN Training")

    for s in bar:
        if use_rad and (s % resample_every == 0) and s > 0:
            key, sub = jax.random.split(key)
            colloc = rad_sample(
                sub,
                problem,
                params,
                static,
                domain=problem.domain,
                n_draw=colloc.shape[0],
                pool_size=pool_size,
                k=k,
                c=c,
            ).astype(jnp.float32)

            if viz_dir and ((s // resample_every) % viz_every == 0):
                _save_colloc_snapshot(colloc, problem.domain, viz_dir, f"step{s}")

        params, opt_state, loss_val = step_fn(params, opt_state, colloc)
        loss_hist.append(float(loss_val))

        if ((s + 1) % eval_every == 0) or (s + 1 == steps):
            rel_l2 = float(eval_fn_rel_l2(params))
            rel_l2_hist.append(rel_l2)
            rel_l2_steps.append(s + 1)
            bar.set_postfix(loss=f"{loss_val:.3e}", relL2=f"{rel_l2:.3e}")
        else:
            bar.set_postfix(loss=f"{loss_val:.3e}")

    return build_model(params), jnp.asarray(loss_hist), (jnp.asarray(rel_l2_steps), jnp.asarray(rel_l2_hist))
