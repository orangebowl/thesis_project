# physics/problems/cosine_ode.py
import jax
import jax.numpy as jnp
from typing import Callable
import equinox as eqx

class PDEProblem:
    domain: tuple # (left, right)

    def ansatz(self, x, nn_out):
        raise NotImplementedError

    def residual(self, model, x):
        raise NotImplementedError

    def exact(self, x):
        return NotImplementedError


class CosineODE(PDEProblem):
    omega  = 1.0
    domain = (jnp.array([-2*jnp.pi]),   
          jnp.array([2*jnp.pi]))   

    # Ansatz (hard contraint)
    @staticmethod
    def ansatz(x, nn_out):
        return jnp.tanh(CosineODE.omega * x) * nn_out 
    # exact solution
    @staticmethod
    def exact(x):
        return jnp.sin(CosineODE.omega * x) / CosineODE.omega

    def _single_res(self, model, x):
        u_x = jax.vmap(jax.grad(lambda y: model(y).squeeze()))(x)
        return jnp.mean((u_x - jnp.cos(self.omega * x))**2)

    # residual
    def residual(self, model, x):
        if isinstance(x, (list, tuple)):                     # FBPINN
            losses = [self._single_res(model, xi) for xi in x]
            return jnp.sum(jnp.stack(losses))
        else:                                                # PINN
            return self._single_res(model, x)




class Poisson2D_freq68(PDEProblem):
    """
    2-D Poisson problem on Ω = [0,1]²
        -Δu = f(x, y),     u|_{∂Ω} = 0
    Exact solution (for testing):
        u(x,y) = sin(6π x²) * sin(8π y²)
    """

    domain = (jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))  # 2D

    @staticmethod
    def exact(xy):
        x, y = xy[..., 0], xy[..., 1]
        return jnp.sin(6 * jnp.pi * x**2) * jnp.sin(8 * jnp.pi * y**2)

    @staticmethod
    def ansatz(xy, nn_out):
        """ Enforces zero boundary conditions: u(x,y) = x(1-x)y(1-y) * NN(x,y). """
        x = xy[..., 0]
        y = xy[..., 1]
        factor = x * (1 - x) * y * (1 - y)  # (N,)
        factor = factor[..., None]          # (N,1)
        return factor * nn_out              # (N,1)

    @staticmethod
    def rhs(xy):
        """ The new right-hand side (source term) f = -Δu. """
        x, y = xy[..., 0], xy[..., 1]
        sin_6pix2 = jnp.sin(6 * jnp.pi * x**2)
        sin_8piy2 = jnp.sin(8 * jnp.pi * y**2)
        cos_6pix2 = jnp.cos(6 * jnp.pi * x**2)
        cos_8piy2 = jnp.cos(8 * jnp.pi * y**2)

        # d^2u/dx^2 term multiplied by sin(8πy^2)
        du_dxx_term = (12 * jnp.pi * cos_6pix2 - 144 * jnp.pi**2 * x**2 * sin_6pix2) * sin_8piy2
        
        # d^2u/dy^2 term multiplied by sin(6πx^2)
        du_dyy_term = (16 * jnp.pi * cos_8piy2 - 256 * jnp.pi**2 * y**2 * sin_8piy2) * sin_6pix2
        
        # f = - (d^2u/dx^2 + d^2u/dy^2)
        return -(du_dxx_term + du_dyy_term)

    def _single_res(self, model, xy_batch):
        """Residual = mean(( -laplacian(u) - f )^2) over xy_batch."""
        if xy_batch.shape[0] == 0:
            return 0.0

        def u_fn(pt_2d):
            out = model(pt_2d)  # shape=(1,1) or scalar
            return out.squeeze()

        hessian_fn = jax.jacfwd(jax.jacrev(u_fn))
        hessians   = jax.vmap(hessian_fn)(xy_batch)
        laplacians = jnp.trace(hessians, axis1=-2, axis2=-1)
        f_vals = self.rhs(xy_batch)
        return jnp.mean( ( -laplacians - f_vals )**2 )

    def residual(self, model, xy):
        """
        Calculates the residual for a single batch.
        Assumes xy is a single batch of shape (N, 2).
        """
        return self._single_res(model, xy)
    
    def pointwise_residual(self, model, xy_batch):
        """
        Calculates the pointwise squared residual: ( -laplacian(u) - f )^2.
        This is needed for residual-based adaptive resampling (RAD).
        """
        if xy_batch.shape[0] == 0:
            return jnp.array([])

        def u_fn(pt_2d):
            out = model(pt_2d)
            return out.squeeze()

        # Calculate the Laplacian for each point in the batch
        hessian_fn = jax.jacfwd(jax.jacrev(u_fn))
        hessians   = jax.vmap(hessian_fn)(xy_batch)
        laplacians = jnp.trace(hessians, axis1=-2, axis2=-1)
        
        # Get the source term values for each point
        f_vals = self.rhs(xy_batch)
        
        # Calculate the squared residual for each point and return the array
        # Note: We do NOT average with jnp.mean() here.
        return ( -laplacians - f_vals )**2



class FirstOrderFreq1010(PDEProblem):
    """
          u_x + u_y = f(x,y)
          u|_{Γ_in} = 0 ,  Γ_in = {(0,y)} ∪ {(x,0)}
    exact  u = sin(10π x²) · sin(10π y²)
    """

    domain = (jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))

    # ---------- exact solution ----------
    @staticmethod
    def exact(xy: jax.Array) -> jax.Array:

        x, y = xy[..., 0], xy[..., 1]
        return jnp.sin(10 * jnp.pi * x**2) * jnp.sin(10 * jnp.pi * y**2)

    # ---------- Dirichlet-0 ansatz with ADF (m=1 soft-min) ----------
    @staticmethod
    def ansatz(xy: jax.Array, nn_out: jax.Array, eps: float = 1e-8) -> jax.Array:
        """
         Γ_in = {x=0} ∪ {y=0}  use soft-min distance instead of x*y：
            phi(x,y) = (x^{-1} + y^{-1})^{-1} = (x*y)/(x+y)   （m=1）
        add eps for stability
        """
        x, y = xy[..., 0], xy[..., 1]
        #x_safe = jnp.maximum(x, eps)
        #y_safe = jnp.maximum(y, eps)
        phi = (x * y) / (x + y + eps)          
        return phi[..., None] * nn_out                        

    #  RHS  f = u_x + u_y 
    @staticmethod
    def rhs(xy: jax.Array) -> jax.Array:
        xy = jnp.atleast_2d(xy)
        x, y = xy[:, 0], xy[:, 1]
        alpha = beta = 10 * jnp.pi
        term_x = 20 * jnp.pi * x * jnp.cos(alpha * x**2) * jnp.sin(beta * y**2)
        term_y = 20 * jnp.pi * y * jnp.sin(alpha * x**2) * jnp.cos(beta * y**2)
        f = term_x + term_y
        return f[:, None]                            # (N,1)

    # ---------- residuals ----------
    def _pointwise_res(self, model: Callable, xy_batch: jax.Array) -> jax.Array:
        xy_batch = jnp.atleast_2d(xy_batch)          # (N,2)
        if xy_batch.shape[0] == 0:
            return jnp.zeros((0, 1))

        # u(x,y)
        def u_fn(pt):
            return model(pt).squeeze()               # scalar

        # ∇u = (u_x, u_y)
        grad_u = jax.vmap(jax.grad(u_fn, argnums=0))(xy_batch)  # (N,2)
        grad_sum = grad_u[:, 0] + grad_u[:, 1]       # (N,)

        r = grad_sum[:, None] - self.rhs(xy_batch)   # (N,1)
        return r

    def _single_res(self, model, xy_batch):
        r = self._pointwise_res(model, xy_batch)
        return jnp.mean(r**2)

    # ---------- FBPINN 接口 ----------
    def pointwise_residual(self, model, xy):
        return self._pointwise_res(model, xy)

    def residual(self, model, xy):
        return self._single_res(model, xy)
    

