import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from typing import Callable, Optional, Dict, Any, Tuple
from tqdm import trange
from pathlib import Path
import matplotlib.pyplot as plt


def _save_colloc_snapshot(colloc: jax.Array, domain: Tuple[jax.Array, jax.Array], out_dir: str, tag: str) -> None:
    """Optionally save a scatter plot of the current collocation set (1D/2D)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    colloc = jnp.asarray(colloc)
    plt.figure(figsize=(6, 4))

    if colloc.shape[1] == 1:
        x = colloc.squeeze()
        plt.scatter(x, jnp.zeros_like(x), s=3, alpha=0.5)
        plt.yticks([])
        plt.xlabel("x")
        plt.xlim(float(domain[0][0]), float(domain[1][0]))
    elif colloc.shape[1] == 2:
        plt.scatter(colloc[:, 0], colloc[:, 1], s=3, alpha=0.5)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(float(domain[0][0]), float(domain[1][0]))
        plt.ylim(float(domain[0][1]), float(domain[1][1]))
    else:
        plt.close()
        return

    plt.title(f"RAD collocation after {tag} (N={colloc.shape[0]})")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / f"rad_{tag}.png", dpi=160)
    plt.close()


@eqx.filter_jit
def pointwise_residual(problem, params, static, x: jax.Array) -> jax.Array:
    """Return per-sample |r(x)| with shape (N,)."""
    model = eqx.combine(params, static)

    def _one(xi):
        # residual expects a batch; add batch dim and take sqrt to get magnitude
        return jnp.sqrt(problem.residual(model, xi[None, ...]))

    return jax.vmap(_one)(x).reshape(-1)


def rad_sample(
    *,
    key: jax.random.PRNGKey,
    problem,
    params,
    static,
    domain: Tuple[jax.Array, jax.Array],
    n_draw: int,
    pool_size: int,
    k: float = 1.0,
    c: float = 1.0,
) -> jax.Array:
    """Residual-based Adaptive Sampling (RAD)."""
    lo, hi = jnp.asarray(domain[0]), jnp.asarray(domain[1])
    dim = lo.size

    # candidate pool
    pool = jax.random.uniform(key, (pool_size, dim), minval=lo, maxval=hi)

    # weights from residuals
    r = pointwise_residual(problem, params, static, pool)  # (pool_size,)
    rpk = r**k
    w = rpk / (jnp.mean(rpk) + 1e-12) + c
    prob = w / (jnp.sum(w) + 1e-12)

    choices = jnp.arange(pool_size)
    if n_draw >= pool_size:
        idx = jax.random.permutation(key, choices)[:n_draw]
    else:
        idx = jax.random.choice(key, choices, shape=(n_draw,), p=prob, replace=False)
    return pool[idx]


def train_pinn(
    *,
    key: jax.random.PRNGKey,
    model: eqx.Module,
    problem,
    colloc: jax.Array,
    lr: float = 3e-4,
    steps: int = 10_000,
    x_test: Optional[jax.Array] = None,
    u_exact: Optional[Callable[[jax.Array], jax.Array]] = None,
    rad_cfg: Optional[Dict[str, Any]] = None,
    eval_every: int = 100,
) -> Tuple[eqx.Module, jax.Array, Tuple[jax.Array, jax.Array]]:
    """
    Full-batch PINN trainer. Tracks PDE loss and relative L2 error on x_test/u_exact.
    Returns (trained_model, loss_history, (eval_steps, rel_l2_history)).
    """
    colloc = colloc.astype(jnp.float32)
    params, static = eqx.partition(model, eqx.is_array)
    build_model = lambda p: eqx.combine(p, static)
    loss_fn = lambda p, xy: problem.residual(build_model(p), xy)

    @eqx.filter_jit
    def eval_fn(p):
        if x_test is None or u_exact is None:
            return jnp.nan
        pred_raw = build_model(p)(x_test.astype(jnp.float32))
        exact_raw = u_exact(x_test)
        pred_vec = pred_raw.ravel()
        exact_vec = exact_raw.ravel()
        if pred_vec.shape != exact_vec.shape:
            return jnp.nan
        err = jnp.linalg.norm(pred_vec - exact_vec)
        den = jnp.linalg.norm(exact_vec)
        return err / (den + 1e-8)

    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @eqx.filter_jit
    def step_fn(p, o, xy_batch):
        xy_batch = xy_batch.reshape(-1, xy_batch.shape[-1])
        loss, grads = jax.value_and_grad(loss_fn)(p, xy_batch)
        updates, o = opt.update(grads, o, p)
        p = eqx.apply_updates(p, updates)
        return p, o, loss

    # ahead-of-time compile
    _ = step_fn(params, opt_state, colloc[:1])

    use_rad = rad_cfg is not None
    if use_rad:
        resample_every = int(rad_cfg.get("resample_every", 1000))
        pool_size = int(rad_cfg.get("pool_size", 10000))
        k = float(rad_cfg.get("k", 1.0))
        c = float(rad_cfg.get("c", 1.0))
        viz_dir = rad_cfg.get("viz_dir", None)
        viz_every = int(rad_cfg.get("viz_every", 1))
        print(f"[RAD] Resampling every {resample_every} steps (pool={pool_size}, k={k}, c={c})")

    loss_hist, rel_l2_hist, eval_steps = [], [], []
    bar = trange(steps, dynamic_ncols=True, desc="PINN Training")

    for s in bar:
        if use_rad and (s % resample_every == 0) and s > 0:
            key, k_rad = jax.random.split(key)
            colloc = rad_sample(
                key=k_rad,
                problem=problem,
                params=params,
                static=static,
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
            rel_l2 = float(eval_fn(params))
            rel_l2_hist.append(rel_l2)
            eval_steps.append(s + 1)
            bar.set_postfix(loss=f"{loss_val:.3e}", RelL2=f"{rel_l2:.3e}")
        else:
            bar.set_postfix(loss=f"{loss_val:.3e}")

    return build_model(params), jnp.asarray(loss_hist), (jnp.asarray(eval_steps), jnp.asarray(rel_l2_hist))
