import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Sequence, Callable


def _init_layer(key: jax.Array,
                in_size: int,
                out_size: int,
                *,
                use_bias: bool = True):
    """Glorot (Xavier) uniform: U[-sqrt(6/(fan_in+fan_out)), +sqrt(6/(fan_in+fan_out))]."""
    wkey, bkey = jax.random.split(key)
    limit = jnp.sqrt(6.0 / (in_size + out_size))

    W = jax.random.uniform(wkey,
                           shape=(in_size, out_size),
                           minval=-limit,
                           maxval=limit)

    b = jnp.zeros((out_size,), dtype=W.dtype) if use_bias else jnp.zeros((out_size,), dtype=W.dtype)
    return W, b

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

        keys = jax.random.split(key, len(layer_sizes) - 1)

        for i in range(len(layer_sizes) - 1):
            in_size, out_size = layer_sizes[i], layer_sizes[i + 1]
            is_last = (i == len(layer_sizes) - 2)

            layer_key = keys[i]
            linear = _init_layer(layer_key, in_size, out_size)

            if is_last:
                self.layers.append(linear)                 # (W,b)
            else:
                self.layers.append((linear, self.activation))

    # forward
    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.atleast_2d(x)

        for linear, act in self.layers[:-1]:
            W, b = linear
            x = act(x @ W + b)

        W_last, b_last = self.layers[-1]
        return x @ W_last + b_last