import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
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

# ―――――――― 1.2  小 ResNet ‒––––––––––––––––––––––––––––––––––––––––––––
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