import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Sequence, Callable

# Helper function to initialize a single layer's parameters.
# This makes the main __init__ method cleaner.
def _init_layer(key: jax.Array,
                in_size: int,
                out_size: int,
                *,
                use_bias: bool = True):
    """Uniform init:  U[ -sqrt(1/in),  sqrt(1/in) ]."""
    wkey, bkey = jax.random.split(key)
    limit = jnp.sqrt(1.0 / in_size)

    # 1. 权重  --->  shape = (in_size, out_size)
    W = jax.random.uniform(wkey,
                           shape=(in_size, out_size),
                           minval=-limit,
                           maxval=limit)

    # 2. 偏置
    if use_bias:
        b = jax.random.uniform(bkey,
                               shape=(out_size,),
                               minval=-limit,
                               maxval=limit)
    else:
        b = jnp.zeros(out_size, dtype=W.dtype)

    return W, b


# ------------------------------------------------------------- FCN
class FCN(eqx.Module):
    """Fully Connected Network — uniform Xavier-like init, Equinox style."""

    layers: list          # [(W,b) , ( (W,b), act ), ...]
    activation: Callable = eqx.static_field()

    def __init__(self,
                 layer_sizes: Sequence[int],
                 key: jax.Array,
                 activation: Callable = jax.nn.tanh):
        """
        Parameters
        ----------
        layer_sizes : e.g. [2, 64, 64, 1]
        key         : JAX PRNGKey
        activation  : hidden-layer non-linearity
        """
        self.activation = activation
        self.layers = []

        # 为每一层切分随机 key
        keys = jax.random.split(key, len(layer_sizes) - 1)

        for i in range(len(layer_sizes) - 1):
            in_size, out_size = layer_sizes[i], layer_sizes[i + 1]
            is_last = (i == len(layer_sizes) - 2)

            # —— 使用统一的 uniform 初始化 —— #
            layer_key = keys[i]
            linear = _init_layer(layer_key, in_size, out_size)

            # 最后一层不加激活
            if is_last:
                self.layers.append(linear)                 # (W,b)
            else:
                self.layers.append((linear, self.activation))

    # --------------------------------------------------------- forward
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass; 支持 (N,d) 或 (d,) 输入。"""
        x = jnp.atleast_2d(x)

        for linear, act in self.layers[:-1]:
            W, b = linear
            x = act(x @ W + b)

        W_last, b_last = self.layers[-1]
        return x @ W_last + b_last