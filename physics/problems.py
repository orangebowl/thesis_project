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
    order  = 1

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
    二阶 ODE:  u''(x) + f(x) = 0 ,  x∈[0, 8]
    解析解   :  u(x) = sin(π x² /4)
    边界条件 :  u(0)=u(8)=0 通过 ansatz 硬约束
    """
    domain = (0.0, 8.0)
    _pi4   = jnp.pi / 4.0

    @staticmethod
    def exact(x):
        return jnp.sin((jnp.pi / 4.0) * x**2)

    # ---------- f(x) ----------
    def _f(self, x):
        return (jnp.pi**2 / 4.0) * x**2 * jnp.sin(self._pi4 * x**2) \
               - (jnp.pi / 2.0) * jnp.cos(self._pi4 * x**2)

    # ---------- ansatz ----------
    def ansatz(self, x, nn_out):
        left, right = self.domain
        A = (1 - jnp.exp(-x)) * (1 - jnp.exp(-(right - x)))
        return A * nn_out

    # ---------- 单子域残差 ----------
    def _single_res(self, model, x):
        def u_scalar(xx):
            return model(xx).squeeze()

        u_xx = jax.vmap(jax.grad(jax.grad(u_scalar)))(x)
        residual = u_xx + self._f(x)
        return jnp.mean(residual**2)

    # ---------- 总残差：兼容 PINN / FBPINN ----------
    def residual(self, model, x):
        if isinstance(x, (list, tuple)):                 # FBPINN: list[ndarray]
            losses = [self._single_res(model, xi) for xi in x]
            return jnp.sum(jnp.stack(losses))
        else:                                            # PINN: ndarray
            return self._single_res(model, x)
