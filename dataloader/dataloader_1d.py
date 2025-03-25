import jax.random as jr
import jax.numpy as jnp


class CollocationData1D:
    def __init__(self, domain=(0.0, 1.0), num_points=1000, seed=0):
        self.domain = domain
        self.num_points = num_points
        self.key = jr.PRNGKey(seed)

    def sample_uniform(self):
        return jr.uniform(self.key, (self.num_points,), minval=self.domain[0], maxval=self.domain[1])

    def sample_grid(self, resolution=200):
        return jnp.linspace(self.domain[0], self.domain[1], resolution)

    def sample_batch(self, batch_size):
        self.key, subkey = jr.split(self.key)
        return jr.uniform(subkey, (batch_size,), minval=self.domain[0], maxval=self.domain[1])