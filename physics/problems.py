# physics/problems/cosine_ode.py
import jax
import jax.numpy as jnp

class PDEProblem:
    domain: tuple # (left, right)

    def ansatz(self, x, nn_out):
        raise NotImplementedError

    def residual(self, model, x):
        raise NotImplementedError

    def exact(self, x):
        return NotImplementedError


class CosineODE(PDEProblem):
    omega  = 15.0
    domain = (-2 * jnp.pi, 2 * jnp.pi)

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
        
class SinPhiPoisson(PDEProblem):
    """
    second order ODE:  u''(x) + f(x) = 0 ,  x∈[0, 8]
    exact solution   :  u(x) = sin(π x² /4)
    boundary constraint :  u(0)=u(8)=0 use ansatz as hard constraint
    """
    domain = (0.0, 8.0)
    _pi4   = jnp.pi / 4.0

    @staticmethod
    def exact(x):
        return jnp.sin((jnp.pi / 4.0) * x**2)

    def _f(self, x):
        return (jnp.pi**2 / 4.0) * x**2 * jnp.sin(self._pi4 * x**2) \
               - (jnp.pi / 2.0) * jnp.cos(self._pi4 * x**2)

    def ansatz(self, x, nn_out):
        left, right = self.domain
        A = (1 - jnp.exp(-x)) * (1 - jnp.exp(-(right - x)))
        return A * nn_out

    # single residual in one subdomain
    def _single_res(self, model, x):
        def u_scalar(xx):
            return model(xx).squeeze()

        u_xx = jax.vmap(jax.grad(jax.grad(u_scalar)))(x)
        residual = u_xx + self._f(x)
        return jnp.mean(residual**2)

    def residual(self, model, x):
        if isinstance(x, (list, tuple)):                 # FBPINN: list[ndarray]
            losses = [self._single_res(model, xi) for xi in x]
            return jnp.sum(jnp.stack(losses))
        else:                                            # PINN: ndarray
            return self._single_res(model, x)


import jax
import jax.numpy as jnp

class SineX6ODE(PDEProblem):
    """
    First-order nonlinear ODE:
        dy/dx = 6x^5 * cos(x^6), with y(0) = 0
    Exact solution:
        y(x) = sin(x^6)
    Domain:
        x ∈ [0, 2]
    """
    domain = (0, 2)
    @staticmethod
    def exact(x):
        return jnp.sin(x**6)

    # Hard constraint: y(0) = 0
    @staticmethod
    def ansatz(x, nn_out):
        return jnp.tanh(x) * nn_out  # Satisfies y(0) = 0

    def _single_res(self, model, x):
        def u_scalar(xx):
            return model(xx).squeeze()

        u_x = jax.vmap(jax.grad(u_scalar))(x)
        # The target now directly uses x and cos(x^4) instead of relying on y
        target = 6 * x**5 * jnp.cos(x**6)
        return jnp.mean((u_x - target)**2)

    def residual(self, model, x):
        if isinstance(x, (list, tuple)):  # FBPINN
            losses = [self._single_res(model, xi) for xi in x]
            return jnp.sum(jnp.stack(losses))
        else:  # PINN
            return self._single_res(model, x)

class Poisson2D(PDEProblem):
    """
    2D Poisson Problem:
        -Δu = f(x, y) in [0,1]^2
        u = 0 on boundary
    Exact: u(x,y) = sin(2πx) * sin(2πy)
    """

    domain = (jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))  # 2D

    @staticmethod
    def exact(xy):
        x, y = xy[..., 0], xy[..., 1]
        return jnp.sin(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y)

    @staticmethod
    def ansatz(xy, nn_out):
        x, y = xy[..., 0], xy[..., 1]
        return x * (1 - x) * y * (1 - y) * nn_out  # zero on ∂Ω

    def _single_res(self, model, xy_batch):
        """Residual for -Δu = f, where xy_batch shape = (N, 2)"""

        def u_fn(xx):  # xx is (2,)
            return model(xx).squeeze()

        # Compute Hessian ∇²u for each point in batch
        hessian_fn = jax.jacfwd(jax.jacrev(u_fn))  # (2,) → (2,2)
        hessians = jax.vmap(hessian_fn)(xy_batch)  # (N,2,2)
        laplacians = jnp.trace(hessians, axis1=-2, axis2=-1)  # (N,)

        x, y = xy_batch[:, 0], xy_batch[:, 1]
        f = -8 * jnp.pi**2 * jnp.sin(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y)  # (N,)

        return jnp.mean((laplacians - f) ** 2)

    def residual(self, model, xy):
        if isinstance(xy, (list, tuple)):
            losses = [self._single_res(model, xi) for xi in xy]
            return jnp.sum(jnp.stack(losses))
        else:
            return self._single_res(model, xy)