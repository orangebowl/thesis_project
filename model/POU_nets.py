import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random

class BasePOUNet:
    num_experts: int
    input_dim:   int
    def init_params(self): ...
    def forward   (self, params, x): ...

def glorot(key, shape):
    fan_in, fan_out = shape
    lim = jnp.sqrt(6. / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-lim, maxval= lim)

# ―――――――― 1.1  MLP ‒––––––––––––––––––––––––––––––––––––––––––––––––––
class MLPPOUNet(BasePOUNet):
    def __init__(self, input_dim: int, num_experts: int,
                 hidden=(32,32,32), key=None):
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

# ------------------------ box initialization ------------------------
def box_init(rng, w_in, w_out, m, delta, L, l):
    """Box initialization (per-column)."""
    k1, k2 = random.split(rng)
    p = m * random.uniform(k1, (w_in, w_out))
    n = random.normal(k2, (w_in, w_out))
    n = n / (jnp.linalg.norm(n, axis=0, keepdims=True) + 1e-9)
    p_max = m * jnp.maximum(0.0, jnp.sign(n))
    k = 1.0 / ((L - 1) * (jnp.sum((p_max - p) * n, axis=0) + 1e-9))
    W = k * n                     # (w_in, w_out)
    b = jnp.sum(W * p, axis=0)    # (w_out,)
    return W, b

# ------------------------ ResNetPOUNet with box_init ----------------
class ResNetPOUNet_Box:
    """
    ResNet-style POU 网络，所有权重采用 box initialization。
      depth : 残差 block 数（每 block 含 2 层）
      width : 隐层宽度
    """
    def __init__(self, input_dim: int, num_experts: int,
                 width=64, depth=4, key=None):
        self.input_dim, self.num_experts = input_dim, num_experts
        if key is None:
            key = random.PRNGKey(42)

        # 计算总层数 L = 1 + 2*depth + 1
        L = 2 * depth + 2
        delta = 1.0 / L           # 与原 box_init 写法一致

        # --------- 1. 输入投影 (idx = 0) ---------
        k0, key = random.split(key)
        W0, b0 = box_init(k0, input_dim, width,
                          m=(1 + delta) ** 1,  # idx+1 = 1
                          delta=delta, L=L, l=1)
        self.p_in = {"W": W0, "b": b0}

        # --------- 2. residual blocks ---------
        self.p_blocks = []
        for blk in range(depth):
            # 两层对应 idx = 1+2*blk, 1+2*blk+1
            k1, key = random.split(key)
            W1, b1 = box_init(k1, width, width,
                              m=(1 + delta) ** (2*blk + 2),
                              delta=delta, L=L, l=2*blk + 2)
            k2, key = random.split(key)
            W2, b2 = box_init(k2, width, width,
                              m=(1 + delta) ** (2*blk + 3),
                              delta=delta, L=L, l=2*blk + 3)
            self.p_blocks.append({"W1": W1, "b1": b1,
                                  "W2": W2, "b2": b2})

        # --------- 3. read-out (idx = L-1) ---------
        kR, key = random.split(key)
        W_out, b_out = box_init(kR, width, num_experts,
                                m=(1 + delta) ** (L),
                                delta=delta, L=L, l=L)
        self.p_out = {"W": W_out, "b": b_out}

    # ---- 复制参数 ----
    def init_params(self):
        return {"p_in": {k: v.copy() for k, v in self.p_in.items()},
                "p_blocks": [{k: v.copy() for k, v in blk.items()}
                             for blk in self.p_blocks],
                "p_out": {k: v.copy() for k, v in self.p_out.items()}}

    # ---- 残差块前向 ----
    @staticmethod
    def _block_forward(p_blk, h):
        y = jax.nn.relu(h @ p_blk["W1"] + p_blk["b1"])
        y = h + (y @ p_blk["W2"] + p_blk["b2"])
        return jax.nn.relu(y)

    # ---- 网络前向 ----
    def forward(self, params, x):
        h = jax.nn.relu(x @ params["p_in"]["W"] + params["p_in"]["b"])
        for blk_p in params["p_blocks"]:
            h = self._block_forward(blk_p, h)
        logits = h @ params["p_out"]["W"] + params["p_out"]["b"]
        return jax.nn.softmax(logits, axis=-1)

# ―――――――― 1.2  小 ResNet ‒––––––––––––––––––––––––––––––––––––––––––––
'''class ResNetPOUNet(BasePOUNet):
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
'''
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