import jax.numpy as jnp
import jax

omega = 15

def f_pde(x):
    return -jnp.cos(omega * x)  # 转为 u'' + f = 0 格式适配 PINN 框架

def u_exact(x):
    return (1.0 / omega) * jnp.sin(omega * x)

def ansatz(x, nn_out):
    """
    tanh(ωx) * NN(x)
    u(0) = 0 
    """
    return jnp.tanh(omega * x) * nn_out

def pde_residual_cosine(model, x, omega=15):
    def u(xx): return model(xx).squeeze()  # 或 model(xx)[0]
    dudx = jax.grad(u)
    return dudx(x) - jnp.cos(omega * x)

pde_problem_cosine = {
    "name": "Cosine",
    "domain": (-6, 6),
    "f_pde": f_pde,
    "u_exact": u_exact,
    "ansatz": ansatz,
    "residual_fn": pde_residual_cosine
}