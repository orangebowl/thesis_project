import jax 
import jax.numpy as jnp
import equinox as eqx
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.window_function import my_window_func


class FBPINN(eqx.Module):
    subnets: tuple
    ansatz: callable = eqx.static_field()
    subdomains: tuple = eqx.static_field()
    num_subdomains: int = eqx.static_field()
    domain: tuple = eqx.static_field()

    def __init__(self, key, num_subdomains, ansatz, subdomains, mlp_config):
        self.ansatz = ansatz
        self.subdomains = subdomains
        self.num_subdomains = num_subdomains
        self.domain = (subdomains[0][0], subdomains[-1][1])  # min(lefts), max(rights)

        keys = jax.random.split(key, num_subdomains)
        self.subnets = tuple(
            eqx.nn.MLP(
                in_size=mlp_config["in_size"],
                out_size=mlp_config["out_size"],
                width_size=mlp_config["width_size"],
                depth=mlp_config["depth"],
                activation=mlp_config["activation"],
                key=k
            )
            for k in keys
        )

    def normalize_x(self, i, x):
        """Normalize input x (n, d) into [-1, 1]^d based on subdomain i."""
        left, right = self.subdomains[i]  # both shape (d,)
        x = jnp.atleast_2d(x)  # shape (n, d)
        center = (left + right) / 2
        scale = (right - left) / 2
        return (x - center) / scale

    def subdomain_pred(self, i, x):
        """Apply subnetwork i to normalized inputs x (n, d)"""
        x = jnp.atleast_2d(x)
        x_norm = self.normalize_x(i, x)  # shape (n, d)
        raw_out = jax.vmap(self.subnets[i])(x_norm)  # (n, 1)
        return raw_out[:, 0]

    def subdomain_window(self, i, x, tol=1e-8):
        """Compute the window weights for subdomain i."""
        x = jnp.atleast_2d(x)
        w_all = my_window_func(self.subdomains, self.num_subdomains, x, tol=tol)  # shape (n, num_sub)
        return w_all[:, i]

    def total_solution(self, x):
        """Return the total solution by summing weighted subnet outputs."""
        x = jnp.atleast_2d(x)
        total = 0.0

        for k in range(self.num_subdomains):
            w = self.subdomain_window(k, x)  # shape (n,)
            out = self.subdomain_pred(k, x)  # shape (n,)
            total += w * out

        return self.ansatz(x, total)

    def __call__(self, x):
        return self.total_solution(x)



# test
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    mlp_config = {
        "in_size": 2,  
        "out_size": 1,  
        "width_size": 8,  
        "depth": 2,  
        "activation": jax.nn.tanh,  
    }
    subdomains = [
    (jnp.array([0.0, 0.0]), jnp.array([0.6, 0.6])),
    (jnp.array([0.4, 0.4]), jnp.array([1.0, 1.0]))
]

    def simple_ansatz(x, total):
        return total

    model = FBPINN(
        key=key,
        num_subdomains=2,
        ansatz=simple_ansatz,
        subdomains=subdomains,
        mlp_config=mlp_config  
    )

    test_x = jnp.array([0.5, 1.5])
    output = model(test_x)
    print("model output:", output)
