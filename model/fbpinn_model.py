import jax
import numpy as np
import jax.numpy as jnp
import equinox as eqx
import os, sys
from typing import Sequence, Tuple, Callable,Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.window_function import cosine,sigmoid
from model.Networks import FCN


class FBPINN(eqx.Module):
    # -------- static fields -------- #
    subnets: tuple
    ansatz: callable        = eqx.static_field()
    xmins_all: jax.Array    = eqx.static_field()
    xmaxs_all: jax.Array    = eqx.static_field()
    wmins_all_fixed: jax.Array = eqx.static_field()
    wmaxs_all_fixed: jax.Array = eqx.static_field()
    num_subdomains: int     = eqx.static_field()
    model_out_size: int     = eqx.static_field()
    xdim: int               = eqx.static_field()

    # ------------------------------------------------------------ #
    def __init__(self, key, subdomains_list, ansatz,
                 mlp_config, fixed_transition_width):

        # -------- 1) 解析网络结构 -------- #
        def _make_layer_sizes(cfg):
            if "layer_sizes" in cfg:                               # 新写法
                return cfg["layer_sizes"], cfg.get("activation", jax.nn.tanh)
            # 兼容旧写法
            in_size  = cfg["in_size"]
            out_size = cfg["out_size"]
            width    = cfg["width_size"]
            depth    = cfg["depth"]
            act      = cfg.get("activation", jax.nn.tanh)
            return [in_size] + [width] * depth + [out_size], act

        layer_sizes, activation = _make_layer_sizes(mlp_config)

        self.ansatz          = ansatz
        self.xdim            = layer_sizes[0]
        self.model_out_size  = layer_sizes[-1]
        self.num_subdomains  = len(subdomains_list)

        # -------- 2) 记录子域边界 -------- #
        if self.num_subdomains == 0:
            self.subnets = tuple()
            pshape = (0, self.xdim)
            self.xmins_all = self.xmaxs_all = jnp.empty(pshape)
            self.wmins_all_fixed = self.wmaxs_all_fixed = jnp.empty(pshape)
        else:
            self.xmins_all = jnp.stack([jnp.asarray(s[0]) for s in subdomains_list])
            self.xmaxs_all = jnp.stack([jnp.asarray(s[1]) for s in subdomains_list])
            self.wmins_all_fixed = jnp.full_like(self.xmins_all, fixed_transition_width)
            self.wmaxs_all_fixed = jnp.full_like(self.xmaxs_all, fixed_transition_width)

            # 每个子域一把 key
            keys = jax.random.split(key, self.num_subdomains)

            # -------- 3) 构建子网 (layer_sizes, key, activation) -------- #
            self.subnets = tuple(
                FCN(layer_sizes, k, activation) for k in keys
            )
    def _normalize_x(self, i, x):
        """Normalizes input 'x' to the [-1, 1] domain for the i-th subnet."""
        center = (self.xmins_all[i] + self.xmaxs_all[i]) / 2.0
        scale = (self.xmaxs_all[i] - self.xmins_all[i]) / 2.0
        return (x - center) / jnp.maximum(scale, 1e-9)

    def __call__(self, x):
        x = jnp.atleast_2d(x)                      # (N, xdim)
        N = x.shape[0]

        if self.num_subdomains == 0:
            return self.ansatz(x, jnp.zeros((N, self.model_out_size), x.dtype))

        # 1) 窗函数权重  (N, K)
        all_w = cosine(
            self.xmins_all, self.xmaxs_all,
            self.wmins_all_fixed, self.wmaxs_all_fixed, x
        )

        # 2) 归一化坐标 (K, N, xdim)
        def _norm_all(k_idx, _):
            return self._normalize_x(k_idx, x)
        x_norm = jax.vmap(_norm_all, in_axes=(0, None))(
            jnp.arange(self.num_subdomains), None
        )

        # 3) 前向：Python 层拼 list，再 stack 到 (K, N, out)
        subnet_out = jnp.stack(
            [net(x_norm[k]) for k, net in enumerate(self.subnets)], axis=0
        )

        # 4) 加权求和 (sum over k)
        total_nn_out = jnp.einsum("nk,kno->no", all_w, subnet_out)

        # 5) Ansatz
        return self.ansatz(x, total_nn_out)






class _DefaultWindow(eqx.Module):
    """A default window function that always returns 1. For n_sub=1 PINN."""
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.ones((x.shape[0], 1))

# --- 2. 用这个修正后的版本替换您原来的 FBPINN_PoU 类 ---
class FBPINN_PoU(eqx.Module):
    """Domain‑decomposed PINN with Partition‑of‑Unity windows (local only)."""

    # ---------- trainable ----------
    # window_fn 不再是 static_field，它包含需要被“看到”的动态模块
    subnets: Tuple[FCN, ...]
    window_fn: eqx.Module 

    # ---------- static fields ------
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
        mlp_config: dict,
        ansatz: Callable,
        residual_fn: Callable,
        window_fn: Optional[Callable] = None,
        *,
        infer_bounds: bool = True,
        window_on_physical: bool = False,
        scan_resolution: int = 1024,
        eps: float = 1e-8,
    ):
        self.ansatz = ansatz
        self.residual_fn = residual_fn

        # ----- handle window function (Corrected) -----
        if window_fn is None:
            if num_subdomains != 1:
                raise ValueError("For num_subdomains > 1 you must supply window_fn.")
            self.window_fn = _DefaultWindow() # 使用健壮的模块替代 lambda
            infer_bounds = False
        else:
            self.window_fn = window_fn

        # ----- store domain info & misc -----
        self.domain_min, self.domain_max = jnp.asarray(domain[0]), jnp.asarray(domain[1])
        self.window_on_physical = window_on_physical
        self.scan_resolution = scan_resolution
        self.eps = eps

        # ----- build sub‑networks -----
        keys = jax.random.split(key, num_subdomains)
        hidden = [mlp_config["width_size"]] * mlp_config["depth"]
        layer_sizes = [mlp_config["in_size"]] + hidden + [mlp_config["out_size"]]
        self.subnets = tuple(FCN(layer_sizes, k, activation=mlp_config["activation"]) for k in keys)

        # ----- local bounds (only if ns>1) -----
        self.xmins_all = None
        self.xmaxs_all = None
        if num_subdomains > 1 and infer_bounds:
            self.xmins_all, self.xmaxs_all = self._infer_subdomain_bounds()
            if self.xmins_all is None:
                raise RuntimeError("Failed to infer sub‑domain bounds – try higher scan_resolution or check window_fn.")

    def _infer_subdomain_bounds(self):
        # This method remains unchanged
        dim = int(self.domain_min.size)
        grids = [jnp.linspace(self.domain_min[d], self.domain_max[d], self.scan_resolution) for d in range(dim)]
        mesh = jnp.stack(jnp.meshgrid(*grids, indexing="ij"), -1)
        flat = mesh.reshape(-1, dim)
        coords_for_window = flat if self.window_on_physical else self._norm_global(flat)
        w = self.window_fn(coords_for_window)
        
        print(f"Window weights: {w.shape}, min: {jnp.min(w)}, max: {jnp.max(w)}")
        
        active = w > self.eps
        xmin_list, xmax_list = [], []
        for i in range(len(self.subnets)):
            pts = flat[active[:, i]]
            if pts.shape[0] == 0:
                return None, None
            xmin_list.append(jnp.min(pts, 0)); xmax_list.append(jnp.max(pts, 0))
        return jnp.stack(xmin_list), jnp.stack(xmax_list)
        
    def _norm_global(self, x):
        return 2.0 * (x - self.domain_min) / (self.domain_max - self.domain_min) - 1.0

    def _norm_all_local(self, x):
        if self.xmins_all is None or self.xmaxs_all is None:
            raise ValueError("Local bounds not initialised.")
        ctr = (self.xmins_all + self.xmaxs_all) / 2.0
        scl = (self.xmaxs_all - self.xmins_all) / 2.0
        return (jnp.expand_dims(x, 0) - ctr[:, None, :]) / jnp.maximum(scl[:, None, :], 1e-9)

    def __call__(self, x: jax.Array):
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
        raw_all = jax.vmap(_apply, in_axes=(0, None, 0))(batched_p, static, x_in_all)
        
        w = self.window_fn(x_for_window)
        w = jnp.swapaxes(w, 0, 1)[..., None]
        u = jnp.sum(raw_all * w, axis=0)
        return self.ansatz(x, u)

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    domain = (jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))

    mlp_cfg = dict(in_size=2, out_size=1, width_size=16, depth=2, activation=jax.nn.tanh)
    ansatz = lambda x, y: y
    residual = lambda m, x: jnp.zeros(())

    # ---- FB‑PINN test (ns=2, local) ----
    window = _make_simple_window(2)
    model_fb = FBPINN_PoU(key, domain, 2, mlp_cfg, ansatz, residual, window)
    x_test = jax.random.uniform(key, (6, 2))
    print("FBPINN y shape:", model_fb(x_test).shape)

    # ---- PINN fallback (ns=1) ----
    model_pinn = FBPINN_PoU(key, domain, 1, mlp_cfg, ansatz, residual)  # window omitted
    print("PINN y shape:", model_pinn(x_test).shape)  # feed 1‑D points for variety


