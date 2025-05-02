import jax
import jax.numpy as jnp
from model.fbpinn_model import FBPINN

# ------------------------------
# PDE & Exact Solution
# ------------------------------
DOMAIN = (0, 8)
LEFT, RIGHT = DOMAIN

def phi(x):
    return (jnp.pi / 4.0) * x**2

def u_exact(x):
    return jnp.sin(phi(x))

def f_pde(x):
    """
    Source function f(x), the PDE is u''(x) + f(x) = 0,
    with the exact solution u(x) = sin(phi(x)).
    """
    return (jnp.pi**2 / 4.0) * x**2 * jnp.sin(phi(x)) - (jnp.pi / 2.0) * jnp.cos(phi(x))

# ------------------------------
# Ansatz Function
# ------------------------------

def ansatz(x, nn_output):
    """
    Used to enforce the boundary conditions u(0) = u(8) = 0.
    We use an exponential function to make the solution decay naturally to 0.
    """
    A_x = (1 - jnp.exp(-x)) * (1 - jnp.exp(-(RIGHT - x)))
    return A_x * nn_output

# ------------------------------
# PDE Residual for PINN
# ------------------------------

def pde_residual_loss(model, subdomain_collocation_points):
    """
    Compute the total loss
    L(θ) = L_p(θ) = (1/N_p) * Σ_i || u''(x_i; θ) + f(x_i) ||²

    where:
    - θ: network parameters
    - u(x_i; θ): predicted solution at point x_i
    - u''(x_i; θ): second derivative of the predicted solution at point x_i
    - f(x_i): source term at x_i
    - N_p: number of collocation points

    This loss minimizes the squared PDE residuals over all collocation points.
    """
    total_loss = 0.0

    # Calculate loss for each subdomain
    for i in range(len(model.subnets)):
        x_i = subdomain_collocation_points[i]
        x_i = jnp.atleast_1d(x_i)  # Ensure x_i is a 1D array

        def u_func(xx):
            # Model solution
            solution = model.total_solution(xx)
            return solution.squeeze()  # Ensure it's a 1D array

        # Second derivative (shape should be (n,))
        u_xx = jax.vmap(lambda xx: jax.grad(jax.grad(u_func))(xx))(x_i)
        #print("Debug:u_xx.shape",u_xx.shape)

        # Compute the PDE residual (u''(x_i) + f(x_i))
        residual = u_xx + f_pde(x_i)  # return (n,)

        # Compute the physical loss for this subdomain
        phys_loss_i = jnp.mean(residual**2)  # Scalar value representing the physical loss for this subdomain

        total_loss += phys_loss_i  # Add up all the losses
    #print("Debug:total_loss.shape",total_loss.shape)
    return total_loss
