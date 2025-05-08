import jax
import equinox as eqx
import jax.numpy as jnp

class PINN(eqx.Module):
    mlp: eqx.nn.MLP
    ansatz: callable = eqx.static_field()

    def __init__(self, key, ansatz, input_dim=1, width=128, depth=5):
        self.mlp = eqx.nn.MLP(
            in_size=input_dim, out_size=1,
            width_size=width, depth=depth,
            activation=jax.nn.tanh, key=key
        )
        self.ansatz = ansatz

    def __call__(self, x):
        x = jnp.atleast_1d(x)
        if x.ndim == 1:
            x_in = x[:, None]
        elif x.ndim == 2:
            x_in = x
        else:
            raise ValueError("Input must be 1D or 2D")
        
        nn_out = self.mlp(x_in)[:, 0]
        return self.ansatz(x, nn_out)
