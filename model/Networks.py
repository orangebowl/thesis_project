# fcn.py
from __future__ import annotations
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import equinox as eqx


# Dense layer
class Dense(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    activation: Callable | None = eqx.static_field()

    def __init__(
        self,
        key: jax.Array,
        in_features: int,
        out_features: int,
        activation: Callable | None = jax.nn.relu,
    ):
        w_key, b_key = jax.random.split(key)
        limit = jnp.sqrt(1.0 / in_features)
        self.weight = jax.random.uniform(
            w_key, (out_features, in_features), minval=-limit, maxval=limit
        )
        self.bias = jax.random.uniform(
            b_key, (out_features,), minval=-limit, maxval=limit
        )
        self.activation = activation

    def __call__(self, x: jax.Array) -> jax.Array:
        y = x @ self.weight.T + self.bias
        return self.activation(y) if self.activation is not None else y


class FCN(eqx.Module):
    """MLP"""
    layers: tuple[Dense, ...]

    def __init__(
        self,
        key: jax.Array,
        in_size: int,
        out_size: int,
        hidden_sizes: Sequence[int] | int = 64,
        activation: Callable = jax.nn.tanh,
        final_activation: Callable | None = None,
    ):
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        keys = jax.random.split(key, len(hidden_sizes) + 1)
        sizes = [in_size, *hidden_sizes, out_size]

        self.layers = tuple(
            Dense(
                k,
                in_features=sizes[i],
                out_features=sizes[i + 1],
                activation=activation if i < len(hidden_sizes) else final_activation,
            )
            for i, k in enumerate(keys)
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.atleast_2d(x)
        for dense in self.layers:
            x = dense(x)
        return x

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)

    # 2→64→64→1
    net = FCN(key, in_size=2, out_size=1, hidden_sizes=[64, 64], activation=jax.nn.tanh)

    # (batch=5, dim=2)
    x_test = jax.random.normal(key, (5, 2))

    # forward
    y_pred = net(x_test)

    print("Input shape :", x_test.shape)   # (5, 2)
    print("Output shape:", y_pred.shape)   # (5, 1)
    print("Sample output:\n", y_pred)
