import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, Callable, Optional

from utils.window_function import cosine
from model.Networks import FCN


class FBPINN(eqx.Module):
    """FBPINN with fixed-width cosine windows and per-subdomain MLPs."""

    # trainable
    subnets: Tuple[FCN, ...]

    # static
    ansatz: Callable = eqx.static_field()
    xmins_all: jax.Array = eqx.static_field()
    xmaxs_all: jax.Array = eqx.static_field()
    wmins_all_fixed: jax.Array = eqx.static_field()
    wmaxs_all_fixed: jax.Array = eqx.static_field()
    num_subdomains: int = eqx.static_field()
    model_out_size: int = eqx.static_field()
    xdim: int = eqx.static_field()

    def __init__(
        self,
        key: jax.Array,
        subdomains_list,
        ansatz: Callable,
        mlp_config: dict,                  # expects: in_size, out_size, width_size, depth, activation
        fixed_transition_width: float,
    ):
        hidden = [mlp_config["width_size"]] * mlp_config["depth"]
        layer_sizes = [mlp_config["in_size"]] + hidden + [mlp_config["out_size"]]
        activation = mlp_config["activation"]

        self.ansatz = ansatz
        self.xdim = layer_sizes[0]
        self.model_out_size = layer_sizes[-1]
        self.num_subdomains = len(subdomains_list)

        if self.num_subdomains == 0:
            self.subnets = tuple()
            empty = jnp.empty((0, self.xdim))
            self.xmins_all = empty
            self.xmaxs_all = empty
            self.wmins_all_fixed = empty
            self.wmaxs_all_fixed = empty
            return

        self.xmins_all = jnp.stack([jnp.asarray(s[0]) for s in subdomains_list])
        self.xmaxs_all = jnp.stack([jnp.asarray(s[1]) for s in subdomains_list])
        self.wmins_all_fixed = jnp.full_like(self.xmins_all, fixed_transition_width)
        self.wmaxs_all_fixed = jnp.full_like(self.xmaxs_all, fixed_transition_width)

        keys = jax.random.split(key, self.num_subdomains)
        self.subnets = tuple(FCN(layer_sizes, k, activation) for k in keys)

    def _normalize_x(self, i: int, x: jax.Array) -> jax.Array:
        """Normalize x into [-1, 1] for the i-th subdomain."""
        center = (self.xmins_all[i] + self.xmaxs_all[i]) / 2.0
        scale = (self.xmaxs_all[i] - self.xmins_all[i]) / 2.0
        return (x - center) / jnp.maximum(scale, 1e-9)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.atleast_2d(x)  # (N, xdim)
        N = x.shape[0]

        if self.num_subdomains == 0:
            return self.ansatz(x, jnp.zeros((N, self.model_out_size), x.dtype))

        # window weights: (N, K)
        all_w = cosine(
            self.xmins_all,
            self.xmaxs_all,
            self.wmins_all_fixed,
            self.wmaxs_all_fixed,
            x,
        )
        all_w = all_w / (jnp.sum(all_w, axis=1, keepdims=True) + 1e-12)

        # normalized inputs per subnet: (K, N, xdim)
        def _norm_all(k_idx, _):
            return self._normalize_x(k_idx, x)

        x_norm = jax.vmap(_norm_all, in_axes=(0, None))(
            jnp.arange(self.num_subdomains), None
        )

        # forward all subnets: (K, N, out)
        subnet_out = jnp.stack(
            [net(x_norm[k]) for k, net in enumerate(self.subnets)], axis=0
        )

        # blend by PoU weights → (N, out)
        total_nn_out = jnp.einsum("nk,kno->no", all_w, subnet_out)

        return self.ansatz(x, total_nn_out)


class _DefaultWindow(eqx.Module):
    """Ones window for the single-subdomain special case."""

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.ones((x.shape[0], 1))


class FBPINN_PoU(eqx.Module):
    """PINN with Partition-of-Unity windows and local subnets."""

    # trainable
    subnets: Tuple[FCN, ...]
    window_fn: eqx.Module

    # static
    ansatz: Callable = eqx.static_field()
    residual_fn: Callable = eqx.static_field()
    domain_min: jax.Array = eqx.static_field()
    domain_max: jax.Array = eqx.static_field()
    xmins_all: Optional[jax.Array] = eqx.static_field()
    xmaxs_all: Optional[jax.Array] = eqx.static_field()
    window_on_physical: bool = eqx.static_field()
    scan_resolution: int = eqx.static_field()
    eps: float = eqx.static_field()

    def __init__(
        self,
        key: jax.Array,
        domain: Tuple[jax.Array, jax.Array],
        num_subdomains: int,
        mlp_config: dict,                 # expects: in_size, out_size, width_size, depth, activation
        ansatz: Callable,
        residual_fn: Callable,
        window_fn: Optional[eqx.Module] = None,
        *,
        infer_bounds: bool = True,
        window_on_physical: bool = False,
        scan_resolution: int = 1024,
        eps: float = 1e-8,
    ):
        self.ansatz = ansatz
        self.residual_fn = residual_fn

        if window_fn is None:
            if num_subdomains != 1:
                raise ValueError("For num_subdomains > 1, a window_fn must be provided.")
            self.window_fn = _DefaultWindow()
            infer_bounds = False
        else:
            self.window_fn = window_fn

        self.domain_min, self.domain_max = jnp.asarray(domain[0]), jnp.asarray(domain[1])
        self.window_on_physical = window_on_physical
        self.scan_resolution = scan_resolution
        self.eps = eps

        hidden = [mlp_config["width_size"]] * mlp_config["depth"]
        layer_sizes = [mlp_config["in_size"]] + hidden + [mlp_config["out_size"]]
        activation = mlp_config["activation"]

        keys = jax.random.split(key, num_subdomains)
        self.subnets = tuple(FCN(layer_sizes, k, activation=activation) for k in keys)

        self.xmins_all = None
        self.xmaxs_all = None
        if num_subdomains > 1 and infer_bounds:
            self.xmins_all, self.xmaxs_all = self._infer_subdomain_bounds()
            if self.xmins_all is None:
                raise RuntimeError(
                    "Failed to infer subdomain bounds. Increase scan_resolution or check window_fn."
                )

    def _infer_subdomain_bounds(self):
        dim = int(self.domain_min.size)
        grids = [jnp.linspace(self.domain_min[d], self.domain_max[d], self.scan_resolution) for d in range(dim)]
        mesh = jnp.stack(jnp.meshgrid(*grids, indexing="ij"), -1)
        flat = mesh.reshape(-1, dim)

        coords = flat if self.window_on_physical else self._norm_global(flat)
        w = self.window_fn(coords)  # (Ngrid, n_sub)

        active = w > self.eps
        xmin_list, xmax_list = [], []
        for i in range(len(self.subnets)):
            pts = flat[active[:, i]]
            if pts.shape[0] == 0:
                return None, None
            xmin_list.append(jnp.min(pts, 0))
            xmax_list.append(jnp.max(pts, 0))
        return jnp.stack(xmin_list), jnp.stack(xmax_list)

    def _norm_global(self, x: jax.Array) -> jax.Array:
        return 2.0 * (x - self.domain_min) / (self.domain_max - self.domain_min) - 1.0

    def _norm_all_local(self, x: jax.Array) -> jax.Array:
        if self.xmins_all is None or self.xmaxs_all is None:
            raise ValueError("Local bounds not initialised.")
        ctr = (self.xmins_all + self.xmaxs_all) / 2.0
        scl = (self.xmaxs_all - self.xmins_all) / 2.0
        return (jnp.expand_dims(x, 0) - ctr[:, None, :]) / jnp.maximum(scl[:, None, :], 1e-9)

    def __call__(self, x: jax.Array) -> jax.Array:
        if x.ndim == 1:
            x = x[None, :]

        ns = len(self.subnets)
        if ns == 1:
            raw = self.subnets[0](x)
            return self.ansatz(x, raw)

        x_in_all = self._norm_all_local(x)
        x_for_window = x if self.window_on_physical else self._norm_global(x)

        p_list = [eqx.partition(n, eqx.is_array)[0] for n in self.subnets]
        static = eqx.partition(self.subnets[0], eqx.is_array)[1]
        batched_p = jax.tree_util.tree_map(lambda *ps: jnp.stack(ps), *p_list)

        def _apply(p, s, xi):
            return eqx.combine(p, s)(xi)

        raw_all = jax.vmap(_apply, in_axes=(0, None, 0))(batched_p, static, x_in_all)  # (n_sub, N, out)

        w = self.window_fn(x_for_window)     # (N, n_sub)
        w = jnp.swapaxes(w, 0, 1)[..., None] # (n_sub, N, 1)
        u = jnp.sum(raw_all * w, axis=0)     # (N, out)

        return self.ansatz(x, u)
