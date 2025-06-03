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
        x ∈ [0, 3]
    """
    domain = (0, 3)
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
        
class SineX5ODE(PDEProblem):
    """
    First-order nonlinear ODE:
        dy/dx = 5x^4 * cos(x^5), with y(0) = 0
    Exact solution:
        y(x) = sin(x^5)
    Domain:
        x ∈ [0, 3]
    """
    domain = (0, 3)
    @staticmethod
    def exact(x):
        return jnp.sin(x**5)

    # Hard constraint: y(0) = 0
    @staticmethod
    def ansatz(x, nn_out):
        return jnp.tanh(x) * nn_out  # Satisfies y(0) = 0

    def _single_res(self, model, x):
        def u_scalar(xx):
            return model(xx).squeeze()

        u_x = jax.vmap(jax.grad(u_scalar))(x)
        # The target now directly uses x and cos(x^4) instead of relying on y
        target = 5 * x**4 * jnp.cos(x**5)
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
        if isinstance(xy, list):
            losses = [self._single_res(model, xi) for xi in xy]
            return jnp.sum(jnp.stack(losses))
        else:
            return self._single_res(model, xy)
        
class Poisson2D_freq(PDEProblem):
    """
    2-D Poisson problem on Ω = [0,1]^2
        -Δu = f(x, y),     u|_{∂Ω} = 0
    Exact solution (for testing):
        u(x,y) = sin(2π x²) * sin(2π y²)
    """

    # 2-D domain 用一对 length-2 的 JAX arrays 表示
    domain = (jnp.array([0.0, 0.0]),   #  x_min = 0 , y_min = 0
          jnp.array([1.0, 1.0]))   #  x_max = 1 , y_max = 1

    @staticmethod
    def exact(xy):
        """
        支持 xy 为 (2,) 或 (...,2)；返回对应形状的 u(x,y)。
        """
        xy = jnp.atleast_1d(xy)         # 处理 0-D 或 (2,) 的情况
        x  = xy[..., 0]
        y  = xy[..., 1]
        return jnp.sin(2 * jnp.pi * x**2) * jnp.sin(2 * jnp.pi * y**2)

    @staticmethod
    def ansatz(xy, nn_out):
        """
        让 u 在边界上自动为 0：factor = x(1-x)*y(1-y)
        支持 xy 为 (2,)、(N,2) 或更高维度 (...,2)。
        """
        xy = jnp.atleast_1d(xy)
        x  = xy[..., 0]
        y  = xy[..., 1]
        factor = x * (1 - x) * y * (1 - y)  # shape (...,)
        factor = factor[..., None]         # shape (...,1)
        return factor * nn_out             # shape (...,1)

    @staticmethod
    def rhs(xy):
        """
        Right-hand side f(x,y) = -Δ u_true(x,y).
        支持 xy 为 (2,) 或 (...,2)。
        """
        xy = jnp.atleast_1d(xy)
        x, y = xy[..., 0], xy[..., 1]
        sin_x2 = jnp.sin(2 * jnp.pi * x**2)
        sin_y2 = jnp.sin(2 * jnp.pi * y**2)
        cos_x2 = jnp.cos(2 * jnp.pi * x**2)
        cos_y2 = jnp.cos(2 * jnp.pi * y**2)
        # 计算 -Δ(sin(2πx²)sin(2πy²))
        term1 = -4 * jnp.pi * (sin_y2 * cos_x2 + sin_x2 * cos_y2)
        term2 = 16 * (jnp.pi**2) * (x**2 + y**2) * sin_x2 * sin_y2
        return term1 + term2  # shape (...,)

    def _single_res(self, model, xy_batch):
        """
        计算 R = mean[ ( -Δu_nn(xy) - f(xy) )^2 ] over xy_batch。
        xy_batch 必须是形如 (N,2)；若传 scalar 或 (2,), 会先 at_least_2d。
        """
        # 保证批量为二维数组 (N,2)，避免后续索引越界
        xy_batch = jnp.atleast_2d(xy_batch)  # shape (N,2)

        if xy_batch.shape[0] == 0:
            return 0.0

        def u_fn(pt_2d):
            """
            单点前向：pt_2d 形如 (2,) 或更低维度，但通过 at_least_1d 强制成 (2,).
            """
            pt_2d = jnp.atleast_1d(pt_2d)
            out = model(pt_2d)         # 可能返回 (1,1) 或 标量
            return out.squeeze()

        # Hessian 操作：对 u_fn 求二阶导
        hessian_fn = jax.jacfwd(jax.jacrev(u_fn))
        # 对整个批量并行计算 Hessian(pt) —— shape=(N,2,2)
        hessians   = jax.vmap(hessian_fn)(xy_batch)
        # 拉普拉斯是 Hessian 对角线之和
        laplacians = jnp.trace(hessians, axis1=-2, axis2=-1)  # shape=(N,)
        f_vals = self.rhs(xy_batch)                           # shape=(N,)

        # 均方残差
        return jnp.mean(( -laplacians - f_vals ) ** 2)

    def residual(self, model, xy):
        """
        统一入口：接受任意形状 xy——如果是标量或 (2,), 会在 _single_res 里自动 at_least_2d。
        如果想做分子域残差，调用者可先分割后传进来。
        """
        return self._single_res(model, xy)

