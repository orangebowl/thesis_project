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
        self.domain = (subdomains[0][0], subdomains[-1][1]) # decided by the left and right
        
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
        """normalize subdomain_i into [-1, 1]"""
        left, right = self.subdomains[i]
        return (x - (left + right) / 2) / ((right - left) / 2)

    '''def unnormalize_x(self, i, x_norm):
        """
        unnormalize subdomain_i from [-1, 1] to original interval
        Not necessary!
        """
        left, right = self.subdomains[i]
        return (x_norm * (right - left) / 2) + (left + right) / 2'''

    def subdomain_pred(self, i, x):
        x = jnp.atleast_1d(x)               # (n,)
        x_norm = self.normalize_x(i, x)          # (n,)
        x_in = x_norm[:, None]                 # (n,1) —— batch DIMENSION
        raw_out = jax.vmap(self.subnets[i])(x_in)
        out = raw_out[:, 0]                   # (n,)
        return out                                

    def subdomain_window(self, i, x, tol=1e-8):
        """Compute the window weights for subdomain i."""
        w_all = my_window_func(self.subdomains, self.num_subdomains, x, tol=tol)  # shape (n, num_sub)
        return w_all[:, i]

    def total_solution(self, x):
        """Return the total solution after combining subdomain outputs."""
        x = jnp.atleast_1d(x)
        total = 0
        n_sub = self.num_subdomains
        
        for k in range(n_sub):
            w = self.subdomain_window(k, x)  # shape (n,)
            #print("Debug:w:",w)
            out = self.subdomain_pred(k, x)  # shape (n,)
            temp = out*w
            total += temp
        total = self.ansatz(x, total)
        # ansatz
        return total
    
    

    def __call__(self, x):
        return self.total_solution(x)


# test
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    mlp_config = {
        "in_size": 1,  
        "out_size": 1,  
        "width_size": 8,  
        "depth": 2,  
        "activation": jax.nn.tanh,  
    }
    subdomains = [(0.0, 1.5), (0.5, 2.0)]

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
