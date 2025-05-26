import jax 
import jax.numpy as jnp
import equinox as eqx
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#from utils.window_function import my_window_func
from utils.window_function import cosine
from model.Networks import FCN

'''class FBPINN(eqx.Module):
    subnets: tuple
    ansatz: callable = eqx.static_field()
    subdomains: tuple = eqx.static_field()
    num_subdomains: int = eqx.static_field()
    domain: tuple = eqx.static_field()
h
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
    '''

class FBPINN(eqx.Module):
    subnets: tuple
    ansatz: callable= eqx.static_field()
    xmins_all: jax.Array= eqx.static_field()
    xmaxs_all: jax.Array= eqx.static_field()
    wmins_all_fixed: jax.Array= eqx.static_field()
    wmaxs_all_fixed: jax.Array= eqx.static_field()
    num_subdomains: int= eqx.static_field()
    xdim: int= eqx.static_field()
    model_out_size: int= eqx.static_field()

    def __init__(self, key, subdomains, ansatz, mlp_config, fixed_transition):
        self.ansatz= ansatz
        self.xdim= mlp_config["in_size"]
        self.model_out_size= mlp_config["out_size"]

        if not subdomains:
            self.num_subdomains=0
            self.subnets= tuple()
            pshape= (0,self.xdim)
            self.xmins_all= jnp.empty(pshape)
            self.xmaxs_all= jnp.empty(pshape)
            self.wmins_all_fixed= jnp.empty(pshape)
            self.wmaxs_all_fixed= jnp.empty(pshape)
        else:
            self.num_subdomains= len(subdomains)
            s_mins= [s[0] for s in subdomains]
            s_maxs= [s[1] for s in subdomains]
            self.xmins_all= jnp.stack(s_mins)
            self.xmaxs_all= jnp.stack(s_maxs)
            self.wmins_all_fixed= jnp.full((self.num_subdomains,self.xdim), fixed_transition)
            self.wmaxs_all_fixed= jnp.full((self.num_subdomains,self.xdim), fixed_transition)

            keys= jax.random.split(key, self.num_subdomains)
            hidden = [mlp_conf["width_size"]] * mlp_conf["depth"]
            self.subnets= tuple(
                FCN(k,
                    in_size= self.xdim,
                    out_size=self.model_out_size,
                    hidden_sizes=hidden,
                    activation= mlp_config["activation"])
                for k in keys
            )

    def _normalize_x(self, i_sub, x):
        left= self.xmins_all[i_sub]
        right= self.xmaxs_all[i_sub]
        center= (left+right)/2.
        scale= (right-left)/2.
        return (x-center)/ jnp.maximum(scale, 1e-9)

    def total_solution(self, x):
        """
        x shape=(N,2).
        We do subdomain-wise MLP + weight, then sum.
        """
        if self.num_subdomains==0:
            return jnp.zeros_like(x[...,0:1])

        w_raw= cosine(self.xmins_all, self.xmaxs_all,
                                      self.wmins_all_fixed, self.wmaxs_all_fixed,
                                      x, tol=1e-8)
        out_list=[]
        for i_sub in range(self.num_subdomains):
            xnorm= self._normalize_x(i_sub, x)
            raw_i= self.subnets[i_sub](xnorm)
            w_i= w_raw[:, i_sub]
            out_i= raw_i*w_i[:,None] if raw_i.ndim==2 else raw_i*w_i
            out_list.append(out_i)
        sum_out= jnp.sum(jnp.stack(out_list,axis=0), axis=0)
        return self.ansatz(x, sum_out)

    def __call__(self, x):
        return self.total_solution(x)

# Example of how to use the modified FBPINN class (minimal)
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    # 
    subdomains = [
        (jnp.array([0.0, 0.0]), jnp.array([0.6, 1.0])),   # left
        (jnp.array([0.4, 0.0]), jnp.array([1.0, 1.0])),   # right
    ]

    # no ansatz
    ansatz = lambda x, nn_out: nn_out
    mlp_conf = dict(in_size=2, out_size=1, width_size=32, depth=2, activation=jax.nn.tanh)
    model = FBPINN(key, subdomains, ansatz, mlp_conf, fixed_transition=0.2)

    N = 8
    x_test = jax.random.uniform(key, (N, 2))  # (N,2) in [0,1]²
    y_pred = model(x_test)

    # 6) 简单断言
    assert y_pred.shape == (N, 1), "should be (N,1)"
    w_sum = cosine(
        model.xmins_all,
        model.xmaxs_all,
        model.wmins_all_fixed,
        model.wmaxs_all_fixed,
        x_test,
    ).sum(axis=1)
    assert jnp.allclose(w_sum, 1.0, rtol=1e-5), "check the weights of window func."

    print("Output sample:\n", y_pred)