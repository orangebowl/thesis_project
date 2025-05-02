# physics/losses.py
import jax
import jax.numpy as jnp
from typing import Sequence
from physics.core import PDEProblem

def _nth_grad(fun, n):
    for _ in range(n):
        fun = jax.grad(fun)
    return fun

def _single_domain_loss(model, x, problem: PDEProblem):
    # u^{(n)}(x)
    def u_scalar(xx):
        return model.total_solution(xx).squeeze()

    u_n = jax.vmap(_nth_grad(u_scalar, problem.order))(x)
    residual = u_n - problem.rhs(x)
    return jnp.mean(residual**2)

def pde_residual_loss(model, collocation, problem: PDEProblem):
    """collocation 可为 ndarray 或 list[ndarray]"""
    if isinstance(collocation, (list, tuple)):
        losses = [_single_domain_loss(model, xi, problem) for xi in collocation]
        return jnp.sum(jnp.stack(losses))
    else:
        return _single_domain_loss(model, collocation, problem)
