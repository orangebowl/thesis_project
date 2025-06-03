# fcn.py
from __future__ import annotations
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import equinox as eqx


# Dense layer

import jax
import jax.numpy as jnp
from jax import random, vmap
from typing import Sequence, Callable


from typing import Sequence, Callable


class FCN(eqx.Module):
    """多层感知机：所有权重都是可训练 PyTree。"""
    weights: list            # [(W,b), ...]
    activation: Callable = eqx.static_field()

    def __init__(self,
                 layer_sizes: Sequence[int],
                 key: jax.Array,
                 activation: Callable = jax.nn.tanh):
        self.activation = activation
        keys = jax.random.split(key, len(layer_sizes) - 1)
        self.weights = []
        for k, (m, n) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
            limit = jnp.sqrt(1.0 / m)
            w_key, b_key = jax.random.split(k)
            W = jax.random.uniform(w_key, (m, n), minval=-limit, maxval=limit)
            b = jax.random.uniform(b_key, (n,),  minval=-limit, maxval=limit)
            self.weights.append((W, b))

    def __call__(self, x: jax.Array) -> jax.Array:     # forward
        x = jnp.atleast_2d(x)
        for W, b in self.weights[:-1]:
            x = self.activation(x @ W + b)
        W_last, b_last = self.weights[-1]
        return x @ W_last + b_last

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)

    # 2→64→64→1
    model = FCN([2, 64, 64, 1], key)

    # (batch=5, dim=2)
    x_test = jax.random.normal(key, (5, 2))

    # forward
    y_pred = model(x_test)

    print("Input shape :", x_test.shape)   # (5, 2)
    print("Output shape:", y_pred.shape)   # (5, 1)
    print("Sample output:\n", y_pred)
