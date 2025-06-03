"""
This file contains RBF/MLP/ResNet for Paritions 
Only use LSGD
"""

from __future__ import annotations
import math, argparse, pathlib, functools, dataclasses
import jax, jax.numpy as jnp, optax, matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------
#  0. 公用工具
# ---------------------------------------------------------------------
SAVE_DIR = pathlib.Path("visualizations")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

def glorot(key, shape):
    fan_in, fan_out = shape
    lim = jnp.sqrt(6. / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-lim, maxval= lim)

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

# ---------------------------------------------------------------------
#  1. Partition Networks
# ---------------------------------------------------------------------
class BasePOUNet:
    num_experts: int
    input_dim:   int
    def init_params(self): ...
    def forward   (self, params, x): ...

# ―――――――― 1.1  MLP ‒––––––––––––––––––––––––––––––––––––––––––––––––––
class MLPPOUNet(BasePOUNet):
    def __init__(self, input_dim: int, num_experts: int,
                 hidden=(32,32,32,32), key=None):
        self.input_dim, self.num_experts = input_dim, num_experts
        key = jax.random.PRNGKey(0) if key is None else key
        keys = jax.random.split(key, len(hidden)+1)
        p, in_dim = {}, input_dim
        for i, h in enumerate(hidden):
            p[f"W{i}"] = glorot(keys[i], (in_dim, h))
            p[f"b{i}"] = jnp.zeros((h,))
            in_dim     = h
        p["W_out"] = glorot(keys[-1], (in_dim, num_experts))
        p["b_out"] = jnp.zeros((num_experts,))
        self._init_params = p

    def init_params(self): return {k: v.copy() for k, v in self._init_params.items()}

    @staticmethod
    def forward(params, x):
        h, n_layer = x, (len(params)//2)-1
        for i in range(n_layer):
            h = jnp.tanh(h @ params[f"W{i}"] + params[f"b{i}"])
        logits = h @ params["W_out"] + params["b_out"]
        return jax.nn.softmax(logits, -1)

# ―――――――― 1.2   ResNet ‒––––––––––––––––––––––––––––––––––––––––––––
class ResNetPOUNet(BasePOUNet):
    def __init__(self, input_dim: int, num_experts: int,
                 width=64, depth=4, key=None):
        """深度=block 数（每个 block 含两层+skip）"""
        self.input_dim, self.num_experts = input_dim, num_experts
        key = jax.random.PRNGKey(42) if key is None else key
        k1, k2 = jax.random.split(key)
        # 输入投影
        self.p_in = {"W": glorot(k1, (input_dim, width)),
                     "b": jnp.zeros((width,))}
        # residual blocks
        keys = jax.random.split(k2, depth*2)
        p_blocks = []
        for i in range(depth):
            W1 = glorot(keys[2*i],   (width, width))
            W2 = glorot(keys[2*i+1], (width, width))
            p_blocks.append({"W1": W1, "W2": W2,
                             "b1": jnp.zeros((width,)),
                             "b2": jnp.zeros((width,))})
        self.p_blocks = p_blocks
        # readout
        kr = jax.random.split(key, 1)[0]
        self.p_out = {"W": glorot(kr, (width, num_experts)),
                      "b": jnp.zeros((num_experts,))}

    def init_params(self):
        return {"p_in": self.p_in,
                "p_blocks": [{k: v.copy() for k, v in blk.items()}
                             for blk in self.p_blocks],
                "p_out": self.p_out}

    @staticmethod
    def _block_forward(p_blk, h):
        y = jax.nn.relu(h @ p_blk["W1"] + p_blk["b1"])
        y = h + (y @ p_blk["W2"] + p_blk["b2"])
        return jax.nn.relu(y)

    def forward(self, params, x):
        h = jax.nn.relu(x @ params["p_in"]["W"] + params["p_in"]["b"])
        for blk_p in params["p_blocks"]:
            h = self._block_forward(blk_p, h)
        logits = h @ params["p_out"]["W"] + params["p_out"]["b"]
        return jax.nn.softmax(logits, -1)

# ―――――――― 1.3  RBF ‒––––––––––––––––––––––––––––––––––––––––––––––––––
class RBFPOUNet(BasePOUNet):
    def __init__(self, input_dim: int, num_centers: int, key=None):
        self.input_dim, self.num_experts = input_dim, num_centers
        key = jax.random.PRNGKey(1) if key is None else key
        k1, k2 = jax.random.split(key)
        base   = jax.random.uniform(k1, (num_centers, input_dim))
        jitter = 0.02 * jax.random.normal(k2, base.shape)
        self._init_centers = jnp.clip(base + jitter, 0., 1.)
        self._init_widths  = 0.15 * jnp.ones((num_centers,))

    def init_params(self):
        return {"centers": self._init_centers.copy(),
                "widths":  self._init_widths.copy()}

    @staticmethod
    def forward(params, x):
        c, w = params["centers"], params["widths"]
        d2   = jnp.sum((x[:,None,:] - c[None,:,:])**2, -1)
        log_phi = -d2 / (w**2 + 1e-12)        # (N,C)
        log_phi = log_phi - jnp.max(log_phi,1,keepdims=True)
        phi = jnp.exp(log_phi)
        return phi / jnp.sum(phi,1,keepdims=True)

# ---------------------------------------------------------------------
#  2. 一阶段  LSGD
# ---------------------------------------------------------------------
@dataclasses.dataclass
class LSGDConfig:
    n_epochs:   int   = 6000
    lr:         float = 1e-3
    lam_init:   float = 1e-3
    rho:        float = 0.99
    n_stag:     int   = 200
    prints:     int   = 10
    viz_int:    int|None = 500   # None=不画

def run_lsgd(model: BasePOUNet, params: dict, x, y, cfg: LSGDConfig):
    lam = jnp.array(cfg.lam_init)
    best, stag = jnp.inf, 0
    log_int = max(1, cfg.n_epochs//cfg.prints)

    @jax.jit
    def loss_fn(p, lam_):
        part   = model.forward(p, x)
        coeffs = fit_local_polynomials(x, y, part, lam_)
        pred   = _predict_from_coeffs(x, coeffs, part)
        return jnp.mean((pred-y)**2)

    valgrad = jax.jit(lambda p, l: jax.value_and_grad(
        lambda pp: loss_fn(pp,l))(p))

    opt = optax.adam(cfg.lr); opt_state = opt.init(params)

    print("⏳ compiling... ", end="", flush=True)
    loss_val, grads = valgrad(params, lam); print("done")

    for ep in range(cfg.n_epochs):
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        loss_val, grads = valgrad(params, lam)

        if cfg.viz_int and ep % cfg.viz_int == 0:
            viz_partitions(model, params, title=f"epoch {ep}")

        if ep % log_int == 0:
            print(f"epoch {ep:6d} | loss {loss_val:.6e} | λ={float(lam):.1e}")

        # early-stop / λ 衰减
        if loss_val < best - 1e-12:
            best, stag = loss_val, 0
        else:
            stag += 1
        if stag > cfg.n_stag:
            lam *= cfg.rho; stag = 0

    return params

def _toy_func(x):
    if x.shape[-1] == 1:
        x1 = x[..., 0]
        return jnp.sin(2 * jnp.pi * x1**2)
    else:
        x1, x2 = x[..., 0], x[..., 1]
        return jnp.sin(2 * jnp.pi * x1**2) * jnp.sin(2 * jnp.pi * x2**2)




if __name__ == "__main__":
    from vis.vis_pou import viz_partitions, viz_final

    # === 简单手动配置 =========================================
    DIM     = 2          # 1  或  2
    NET_TAG = "mlp"      # "mlp" | "resnet" | "rbf"
    EPOCHS  = 1000
    N_SUB = 4
    # ==========================================================

    key = jax.random.PRNGKey(0)

    # generate training data
    if DIM == 2:
        xs        = jnp.linspace(0, 1, 40)
        xx, yy    = jnp.meshgrid(xs, xs)
        x_train   = jnp.stack([xx.ravel(), yy.ravel()], -1)   # (1600, 2)
    else:                       # 1-D
        x_train   = jnp.linspace(0, 1, 400)[:, None]          # (400, 1)

    y_train = _toy_func(x_train)

    # Choose for Parition mlp/resnet/rbf
    if NET_TAG == "mlp":
        net = MLPPOUNet(DIM, num_experts=N_SUB, key=key)
    elif NET_TAG == "resnet":
        net = ResNetPOUNet(DIM, num_experts=N_SUB, key=key)
    else:                       # "rbf"
        net = RBFPOUNet(DIM, num_centers=N_SUB, key=key)

    # LSGD 
    params = net.init_params()
    cfg    = LSGDConfig(n_epochs=EPOCHS,
                        viz_int=500 if DIM == 2 else None)
    params = run_lsgd(net, params, x_train, y_train, cfg)

    # vis
    viz_partitions(net, params, title="final partitions")
    viz_final(net, params, x_train, y_train)

    print(f"Done. Images saved to “{SAVE_DIR.absolute()}”.")
