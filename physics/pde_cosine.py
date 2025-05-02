# pde_cosine.py

import jax
import jax.numpy as jnp
from model.fbpinn_model import FBPINN
"""
ODE:
u'(x) - cos(omega * x) = 0
Exact solution: 
u_exact(x) = (1/omega) sin(omega*x)
"""

omega = 15
DOMAIN = (-2 * jnp.pi, 2 * jnp.pi)

def f_ode(x):
    """RHS: cos(omega*x)"""
    return jnp.cos(omega * x)


def u_exact(x):
    """Exact solution: (1/omega)*sin(omega*x)"""
    return (1.0 / omega) * jnp.sin(omega * x)


def ansatz(x, nn_out):
    #print("[Debug]:x.shape:",x.shape)
    """
    tanh(omega*x)*nn_out
    make sure when x=0, output=0。
    """
    nn_out = jnp.tanh(omega*x) * nn_out
    return nn_out
# ---------- 单子域残差 ----------
def _single_domain_loss(model, x):
    """
    x : (Ni,) collocation points of **one** sub-domain  (or whole domain for PINN)
    """
    # u'(x)
    def u_func(xx):               # xx: scalar
        return model.total_solution(xx).squeeze()

    dudx = jax.vmap(jax.grad(u_func))(x)
    residual = dudx - jnp.cos(omega * x)
    return jnp.mean(residual**2)


# ---------- 通用残差 ----------
def pde_residual_loss(model, collocation):
    """
    collocation:
      • ndarray            -> 普通 PINN
      • list/tuple[ndarray] -> FBPINN
    """
    if isinstance(collocation, (list, tuple)):
        # --- FBPINN: 逐子域累加 ---
        losses = [_single_domain_loss(model, x_i) for x_i in collocation]
        return jnp.sum(jnp.stack(losses))
    else:
        # --- 普通 PINN ---
        return _single_domain_loss(model, collocation)

"""def pde_residual_loss(model, x_collocation):
    
    #x_collocation : (N,)  – 整批一次算；不需要再 vmap 外层
    
    # u'(x)
    def u_func(xx):           # xx 是标量
        return model.total_solution(xx).squeeze()

    # d/dx u
    dudx_all = jax.vmap(jax.grad(u_func))(x_collocation)

    residual  = dudx_all - jnp.cos(omega * x_collocation)
    return jnp.mean(residual**2)"""
