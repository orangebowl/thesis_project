import jax
import jax.numpy as jnp
import equinox as eqx
import jax.random as jr
from physics.pde_1d import ansatz  

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def sigmoid_window_function(x, subdomains, sigma):
    x = jnp.atleast_1d(x)
    subdomains = jnp.array(subdomains)
    left = sigmoid((x[:, None] - subdomains[:, 0]) / sigma)
    right = sigmoid((subdomains[:, 1] - x[:, None]) / sigma)
    w = left * right
    sum_w = jnp.sum(w, axis=1, keepdims=True) + 1e-10
    return w / sum_w

class SmoothFBPINN(eqx.Module):
    subnets: tuple
    subdomains: jnp.ndarray
    sigma: float

    def __init__(self, subdomains, sigma, key):
        self.subdomains = jnp.array(subdomains)
        self.sigma = sigma
        keys = jr.split(key, len(subdomains))
        self.subnets = tuple(
            eqx.nn.MLP(1, 1, width_size=20, depth=3, activation=jax.nn.tanh, key=k)
            for k in keys
        )

    def __call__(self, x):
        x = jnp.atleast_1d(x)
        w = sigmoid_window_function(x, self.subdomains, self.sigma)
        subvals = jnp.array([net(x.reshape((1,)))[0] for net in self.subnets])
        net_sum = jnp.dot(w[0, :], subvals)
        return ansatz(x, net_sum)
