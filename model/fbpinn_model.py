import jax
import jax.numpy as jnp
import equinox as eqx
import os, sys
from typing import Sequence, Tuple, Callable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.window_function import cosine,sigmoid
from model.Networks import FCN


class FBPINN(eqx.Module):
    """Domain‐decomposed PINN supporting arbitrary dimensions."""

    # -------- 可训练参数 --------
    subnets: Tuple[FCN, ...]

    # -------- 静态字段 --------
    ansatz: Callable             = eqx.static_field()
    residual_fn: Callable        = eqx.static_field()
    xmins_all: jax.Array         = eqx.static_field()
    xmaxs_all: jax.Array         = eqx.static_field()
    wmins_fixed: jax.Array       = eqx.static_field()
    wmaxs_fixed: jax.Array       = eqx.static_field()
    xdim: int                    = eqx.static_field()
    window_fn: Callable          = eqx.static_field()   # ← 新增

    # ------------------------------------------------------------------ #
    #                             初始化                                  #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        key: jax.Array,
        subdomains: Sequence[Tuple[jax.Array, jax.Array]],
        mlp_config: dict,
        ansatz: Callable,
        residual_fn: Callable,
        *,
        fixed_transition: float  = 0.2,
        window_fn: Callable  = None,          # ← 可选外部权重函数
    ):
        self.ansatz      = ansatz
        self.residual_fn = residual_fn
        self.xdim        = mlp_config["in_size"]
        self.window_fn   = window_fn               # None -> default window function
        # ---------- 子区间数据 ----------
        self.xmins_all = jnp.stack([s[0] for s in subdomains])
        self.xmaxs_all = jnp.stack([s[1] for s in subdomains])
        if fixed_transition is None:
            # 设成零长度，cosine() 仍可调用但 w≈δ(i)
            fixed_transition = 0.0
        self.wmins_fixed = jnp.full_like(self.xmins_all, fixed_transition)
        self.wmaxs_fixed = jnp.full_like(self.xmins_all, fixed_transition)

        # ---------- 构建子网 ----------
        keys = jax.random.split(key, len(subdomains))
        hidden = [mlp_config["width_size"]] * mlp_config["depth"]
        layer_sizes = [self.xdim] + hidden + [mlp_config["out_size"]]
        self.subnets = tuple(
            FCN(layer_sizes, k, activation=mlp_config["activation"]) for k in keys
            
        )

    
    # Normalised to [-1,1]                           
    def _norm_all(self, x: jax.Array) -> jax.Array:
        left, right = self.xmins_all, self.xmaxs_all              # (ns,d)
        center = (left + right) / 2.0
        scale  = (right - left) / 2.0
        x_tile = jnp.expand_dims(x, 0)                            # (1,N,d)
        return (x_tile - center[:, None, :]) / jnp.maximum(scale[:, None, :], 1e-9)
    '''
    # Normalised to [0,1]  
    def _norm_all(self, x: jax.Array) -> jax.Array:
        """
        把物理坐标 (N,d) 变成 (ns,N,d) ，各子域局部坐标∈[0,1].
        """
        left, right = self.xmins_all, self.xmaxs_all          # (ns,d)
        scale = jnp.maximum(right - left, 1e-9)
        x_tile = x[None, ...]                                 # (1,N,d)
        return (x_tile - left[:, None, :]) / scale[:, None, :]  # ∈[0,1]
    '''
    @eqx.filter_jit
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        x : (N, xdim)
        return u(x): shape (N, out_dim)
        """
        if x.ndim == 1:
            x = x[None, :]

        # normalize
        xnorm_all = self._norm_all(x)                              # (ns,N,d)

        # forward
        raw_all = jnp.stack([net(xn) for net, xn in zip(self.subnets, xnorm_all)], 0)

        # window weights
        if self.window_fn is None:
            # use default cosine window function
            w = cosine(
                self.xmins_all, self.xmaxs_all,
                self.wmins_fixed, self.wmaxs_fixed,
                x
            )                               # (N, ns)
            w = jnp.swapaxes(w, 0, 1)[..., None]     # (ns,N,1)
        else:
            # learned PartitionNet
            w_full = self.window_fn(x)                # (N, ns)
            w = jnp.swapaxes(w_full, 0, 1)[..., None] # (ns,N,1)

        # Ansatz
        sum_out = (raw_all * w).sum(axis=0)            # (N,out)
        return self.ansatz(x, sum_out)

    #Loss                                    
    @eqx.filter_jit
    def loss(self, x_batch: jax.Array) -> jax.Array:
        res_sq = jax.vmap(self.residual_fn, (None, 0))(self, x_batch)
        return jnp.mean(res_sq)

if __name__ == "__main__":                       
    key = jax.random.PRNGKey(0)
    subdomains = [
        (jnp.array([0.0, 0.0]), jnp.array([0.6, 1.0])),
        (jnp.array([0.4, 0.0]), jnp.array([1.0, 1.0])),
    ]
    ansatz = lambda x, y: y
    mlp_conf = dict(in_size=2, out_size=1, width_size=32, depth=2,
                    activation=jax.nn.tanh)

    model = FBPINN(
        key          = key,
        subdomains   = subdomains,
        mlp_config   = mlp_conf,
        ansatz       = ansatz,
        residual_fn  = lambda m, x: jnp.sum(x),   # dummy
        fixed_transition = 0.2,
        window_fn    = None                       # 后续可换成 PartitionNet
    )

    N = 8
    x_test = jax.random.uniform(key, (N, 2))
    y_pred = model(x_test)
    assert y_pred.shape == (N, 1)
    print("FBPINN forward OK, sample output:\n", y_pred)
