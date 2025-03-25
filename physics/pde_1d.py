# physics/pde_1d.py

import jax
import jax.numpy as jnp


# --- PDE 1 示例 ---
def phi_1(x):
    return (jnp.pi / 4.0) * x**2

def u_exact_1(x):
    return jnp.sin(phi_1(x))

def f_pde_1(x):
    return (jnp.pi**2 / 4.0) * x**2 * jnp.sin(phi_1(x)) - (jnp.pi / 2.0) * jnp.cos(phi_1(x))

def ansatz(x, nn_out):
    A = (1 - jnp.exp(-x)) * (1 - jnp.exp(-(8 - x)))
    return A * nn_out

def pde_residual_1(model, x):
    def u(x_): return model(x_)
    dudx = jax.grad(u)
    d2udx2 = jax.grad(dudx)
    return d2udx2(x) + f_pde_1(x)


# --- PDE 模块接口 ---
pde_problem_1 = {
    "u_exact": u_exact_1,
    "f": f_pde_1,
    "ansatz": ansatz,
    "residual_fn": pde_residual_1,
    "domain": (0.0, 8.0)
}
