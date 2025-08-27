import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable
from model.Networks import FCN 

class PINN(eqx.Module):
    net: FCN
    ansatz: Callable = eqx.static_field()
    in_dim: int = eqx.static_field()
    out_dim: int = eqx.static_field()
    lo: jax.Array = eqx.static_field()
    hi: jax.Array = eqx.static_field()

    def __init__(self, *, key: jax.Array, ansatz: Callable, mlp_config: dict,
                 domain: tuple[jax.Array, jax.Array]):
        in_size, out_size = mlp_config["in_size"], mlp_config["out_size"]
        width_size, depth = mlp_config["width_size"], mlp_config["depth"]
        activation = mlp_config["activation"]

        self.in_dim, self.out_dim = in_size, out_size
        self.ansatz = ansatz
        self.lo, self.hi = domain  # shape = (D,)

        layer_sizes = [in_size, *[width_size] * depth, out_size]
        self.net = FCN(layer_sizes=layer_sizes, key=key, activation=activation)

    def _normalize(self, x):
        # map [lo,hi] -> [-1,1]
        return 2.0 * (x - self.lo) / (self.hi - self.lo) - 1.0

    def __call__(self, x):
        x_reshaped = jnp.atleast_2d(x)
        x_hat = self._normalize(x_reshaped)       
        nn_out = self.net(x_hat)                  #(N, 1)
        return self.ansatz(x_reshaped, nn_out)    