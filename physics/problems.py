# physics/problems/cosine_ode.py
import jax
import jax.numpy as jnp
from typing import Union, Sequence
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

class SinPhiPoissonSoft(PDEProblem):
    """
    u''(x) + f(x) = 0  ,  x ∈ [0, 8]
    exact         : u(x) = sin(π x² / 4)
    boundary      : u(0)=u(8)=0  —— 这里用“软约束”边界项
    """
    domain = (jnp.array([0]),   
          jnp.array([8]))           # 用纯 float 简化

    _pi4 = jnp.pi / 4.0

    # ---------------- exact solution & forcing ---------------- #
    @staticmethod
    def exact(x):
        return jnp.sin((jnp.pi / 4.0) * x**2)
    
    @staticmethod
    def ansatz(x, nn_out):
        return nn_out                     # 不做任何边界硬约束

    def _f(self, x):
        return (jnp.pi**2 / 4.0) * x**2 * jnp.sin(self._pi4 * x**2) \
               - (jnp.pi / 2.0) * jnp.cos(self._pi4 * x**2)

    # ---------------- interior / PDE residual ----------------- #
    def _pde_res(self, model, x):
        """PDE loss on interior collocation points x (shape (N,) or list[...] )"""
        def _single(xi):                       # xi shape (M,)
            xi = jnp.ravel(xi)                 # 保证 1-D
            def u_scalar(xx):                  # xx → scalar
                return model(xx).reshape(())
            u_xx = jax.vmap(jax.grad(jax.grad(u_scalar)))(xi)
            return jnp.mean((u_xx + self._f(xi))**2)

        if isinstance(x, (list, tuple)):       # FBPINN 分区
            return jnp.sum(jnp.stack([_single(xi) for xi in x]))
        else:                                   # 普通 PINN
            return _single(x)

    # ---------------- boundary residual ----------------------- #
    def _bdy_res(self, model, xb):
        """Boundary loss on xb (shape (Nb,))"""
        u_bdy = jax.vmap(lambda xx: model(xx).reshape(()))(xb)
        print(u_bdy.shape)
        return jnp.mean(u_bdy**2)               # g(x)=0

    # ---------------- combined loss --------------------------- #
    def residual(
        self,
        model,
        x_int,            # interior collocation points
        x_bdy,            # boundary points
        w_pde = 1.0,      # weight for PDE loss
        w_bdy = 1.0,      # weight for boundary loss
    ):
        """
        Returns: total_loss = w_pde * L_pde + w_bdy * L_bdy
        """
        L_pde = self._pde_res(model, x_int)
        L_bdy = self._bdy_res(model, x_bdy)
        return w_pde * L_pde + w_bdy * L_bdy


class SinPhiPoisson(PDEProblem):
    """
    second order ODE:  u''(x) + f(x) = 0 ,  x∈[0, 8]
    exact solution   :  u(x) = sin(α π x²)
    boundary constraint :  u(0)=u(8)=0 use ansatz as hard constraint
    """
    domain = (jnp.array([0.0]), jnp.array([8.0])) 
    alpha = 1.0  # <-- 可调参数，原来是 0.25

    @staticmethod
    def exact(x):
        return jnp.sin(SinPhiPoisson.alpha * jnp.pi * x**2)

    def _f(self, x):
        a = self.alpha
        π = jnp.pi
        return 4 * a**2 * π**2 * x**2 * jnp.sin(a * π * x**2) \
               - 2 * a * π * jnp.cos(a * π * x**2)

    @staticmethod
    def ansatz(x, nn_out):
        left, right = SinPhiPoisson.domain
        return (x - left.item()) * (right.item() - x) * nn_out

    def _single_res(self, model, x):
        x = jnp.ravel(x)
        def u_scalar(xx): return jnp.sum(model(xx))
        u_xx = jax.vmap(jax.grad(jax.grad(u_scalar)))(x)
        res  = u_xx + self._f(x)
        return jnp.mean(res**2)

    def residual(self, model, x):
        if isinstance(x, (list, tuple)):
            losses = [self._single_res(model, xi) for xi in x]
            return jnp.sum(jnp.stack(losses))
        else:
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
        x ∈ [-2, 2]
    """
    domain = (jnp.array([-2]),   
          jnp.array([2])) 
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

class SineX6ODESoft(PDEProblem):
    r"""
    一阶非线性 ODE：
        u'(x) = 6 x^5 · cos(x^6)，  x ∈ [0, 2]
    边界条件：
        u(0) = 0               （这里只对 x=0 施加软约束）
    精确解：
        u(x) = sin(x^6)
    """
    domain = (jnp.array([0.0]), jnp.array([2.0]))

    # ---------------- exact solution ------------------------- #
    @staticmethod
    def exact(x):
        return jnp.sin(x**6)

    # 不再用硬约束 —— ansatz 恒等
    @staticmethod
    def ansatz(x, nn_out):
        return nn_out

    # ---------------- PDE residual (interior) ---------------- #
    def _pde_res(self, model, x):
        """
        L_pde = ⟨[u'(x) − 6 x^5 cos(x^6)]²⟩  (mean over interior points)
        x : ndarray shape (N,)  或 FBPINN 的 list[ndarray]
        """
        def _single(xi):                      # xi shape (M,)
            xi = jnp.ravel(xi)                # 保证 1-D
            # 标量输出的 u(x)
            def u_scalar(xx):
                return model(xx).reshape(())
            # 一阶导
            u_x = jax.vmap(jax.grad(u_scalar))(xi)
            target = 6.0 * xi**5 * jnp.cos(xi**6)
            return jnp.mean((u_x - target)**2)

        if isinstance(x, (list, tuple)):      # FBPINN
            return jnp.sum(jnp.stack([_single(xi) for xi in x]))
        else:                                 # PINN
            return _single(x)

    # ---------------- Boundary residual ---------------------- #
    def _bdy_res(self, model, xb):
        """
        L_bdy = ⟨u(xb)²⟩  (均值)；此处 xb 只需包含 0.0
        xb : ndarray shape (Nb,)
        """
        u_bdy = jax.vmap(lambda xx: model(xx).reshape(()))(xb)
        return jnp.mean(u_bdy**2)             # g(x) = 0

    # ---------------- Combined loss -------------------------- #
    def residual(
        self,
        model,
        x_int,             # interior collocation points
        x_bdy,             # boundary points (建议 [0.0])
        w_pde = 1.0,
        w_bdy = 1.0,
    ):
        """
        total_loss = w_pde * L_pde  +  w_bdy * L_bdy
        """
        L_pde = self._pde_res(model, x_int)
        L_bdy = self._bdy_res(model, x_bdy)
        return w_pde * L_pde + w_bdy * L_bdy


class SineX3ODE(PDEProblem):
    """
    First-order nonlinear ODE:
        dy/dx = 3x^2 * cos(x^3), with y(0) = 0
    Exact solution:
        y(x) = sin(x^)
    Domain:
        x ∈ [-2pi, 2pi]
    """
    domain = (jnp.array([-2*jnp.pi]), jnp.array([2*jnp.pi]))
    @staticmethod
    def exact(x):
        return jnp.sin(x**3)

    # Hard constraint: y(0) = 0
    @staticmethod
    def ansatz(x, nn_out):
        return jnp.tanh(x) * nn_out  # Satisfies y(0) = 0

    def _single_res(self, model, x):
        def u_scalar(xx):
            return model(xx).squeeze()

        u_x = jax.vmap(jax.grad(u_scalar))(x)
        # The target now directly uses x and cos(x^4) instead of relying on y
        target = 3 * x**2 * jnp.cos(x**3)
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
    
    def _pointwise_res(self, model, xy_batch):
        """
        Parameters
        ----------
        model : callable
        xy_batch : (N,2) or (...,2)  collocation points
        Returns
        -------
        r : (N,)  逐点 PDE 残差，**未平方/未平均**
        """
        xy_batch = jnp.atleast_2d(xy_batch)                 # (N,2)
        if xy_batch.shape[0] == 0:
            return jnp.zeros((0,))

        # 单点前向
        def u_fn(pt):
            return model(pt).squeeze()                      # scalar

        # Hessian & Laplacian
        hessian_fn = jax.hessian(u_fn)                      # 2×2
        laplacians = jax.vmap(lambda pt: jnp.trace(hessian_fn(pt)))(xy_batch)
        r = -laplacians - self.rhs(xy_batch)                # (N,)
        return r

    # =================================================================
    #  兼容旧接口：仍然提供 mean-squared residual（标量）
    # =================================================================
    def _single_res(self, model, xy_batch):
        r = self._pointwise_res(model, xy_batch)            # (N,)
        return jnp.mean(r**2)

    # =================================================================
    #  **供 FBPINN 使用的入口**：返回逐点残差向量
    # =================================================================
    def pointwise_residual(self, model, xy):
        """
        与旧 residual 接口并列存在，只做逐点残差。
        """
        return self._pointwise_res(model, xy)

    # 原来的 residual() 保留给其它脚本
    def residual(self, model, xy):
        return self._single_res(model, xy)

class Poisson2D_freq66(PDEProblem):
    """
    2-D Poisson problem on Ω = [0,1]²
        -Δu = f(x, y), u|_{∂Ω} = 0
    Exact solution (for testing):
        u(x,y) = sin(6π x²) · sin(6π y²)
    """
    # ────────────────────────────── 基本信息 ──────────────────────────────
    domain = (jnp.array([0.0, 0.0]),   # x_min, y_min
              jnp.array([1.0, 1.0]))   # x_max, y_max

    # ---------------------------------------------------------------------
    # 真解
    # ---------------------------------------------------------------------
    @staticmethod
    def exact(xy):
        """支持 xy 为 (2,) 或 (...,2)。"""
        xy = jnp.atleast_1d(xy)
        x, y = xy[..., 0], xy[..., 1]
        return jnp.sin(6 * jnp.pi * x**2) * jnp.sin(6 * jnp.pi * y**2)

    # ---------------------------------------------------------------------
    # Ansatz：自动满足零 Dirichlet
    # ---------------------------------------------------------------------
    @staticmethod
    def ansatz(xy, nn_out):
        xy = jnp.atleast_1d(xy)
        x, y   = xy[..., 0], xy[..., 1]
        factor = x * (1 - x) * y * (1 - y)        # (...,)
        return (factor[..., None]) * nn_out       # (...,1)

    # ---------------------------------------------------------------------
    # 右端项 f(x,y) = -Δu_true(x,y)
    # ---------------------------------------------------------------------
    @staticmethod
    def rhs(xy):
        xy = jnp.atleast_1d(xy)
        x, y = xy[..., 0], xy[..., 1]

        sin_6x = jnp.sin(6 * jnp.pi * x**2)
        cos_6x = jnp.cos(6 * jnp.pi * x**2)
        sin_6y = jnp.sin(6 * jnp.pi * y**2)
        cos_6y = jnp.cos(6 * jnp.pi * y**2)

        # ∂²u/∂x²
        d2u_dx2 = (12 * jnp.pi * cos_6x - 144 * (jnp.pi**2) * x**2 * sin_6x) * sin_6y
        # ∂²u/∂y²
        d2u_dy2 = (12 * jnp.pi * cos_6y - 144 * (jnp.pi**2) * y**2 * sin_6y) * sin_6x

        return -(d2u_dx2 + d2u_dy2)               # (...,)

    # =========================================================================
    #  内部：逐点残差  r(x,y) = -Δu_hat − f
    # =========================================================================
    def _pointwise_res(self, model, xy_batch):
        xy_batch = jnp.atleast_2d(xy_batch)          # (N,2)
        if xy_batch.shape[0] == 0:
            return jnp.zeros((0,))

        def u_fn(pt):
            return model(pt).squeeze()               # scalar

        hess_fn   = jax.hessian(u_fn)                # 2×2
        laplacian = jax.vmap(lambda pt: jnp.trace(hess_fn(pt)))(xy_batch)  # (N,)
        r = -laplacian - self.rhs(xy_batch)          # (N,)
        return r

    # =========================================================================
    #  标量残差：mean-squared（兼容旧接口）
    # =========================================================================
    def _single_res(self, model, xy_batch):
        r = self._pointwise_res(model, xy_batch)
        return jnp.mean(r**2)

    # =========================================================================
    #  对外接口
    # =========================================================================
    def residual(self, model, xy):
        """返回 mean-squared residual（标量）。"""
        return self._single_res(model, xy)

    def pointwise_residual(self, model, xy):
        """返回逐点残差向量。"""
        return self._pointwise_res(model, xy)



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
        """ The new exact solution. """
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

from typing import Callable
class FirstOrderFreq68(PDEProblem):
    """
    一阶 PDE (使用手动微分优化)
    """

    domain = (jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))

    # ---------- exact solution (unchanged) ----------
    @staticmethod
    def exact(xy: jax.Array) -> jax.Array:
        xy = jnp.atleast_1d(xy)
        x, y = xy[..., 0], xy[..., 1]
        return jnp.sin(6*jnp.pi*x**2) * jnp.sin(8*jnp.pi*y**2)

    # ---------- ansatz (unchanged) ----------
    @staticmethod
    def ansatz(xy: jax.Array, nn_out: jax.Array) -> jax.Array:
        x, y = xy[..., 0], xy[..., 1]
        factor = x*y
        return factor[..., None] * nn_out

    # ---------- RHS (unchanged) ----------
    @staticmethod
    def rhs(xy: jax.Array) -> jax.Array:
        x, y = xy[..., 0], xy[..., 1]
        α, β = 6*jnp.pi, 8*jnp.pi
        term_x = 12*jnp.pi * x * jnp.cos(α*x**2) * jnp.sin(β*y**2)
        term_y = 16*jnp.pi * y * jnp.sin(α*x**2) * jnp.cos(β*y**2)
        return term_x + term_y

    def _pointwise_res(self, model: eqx.Module, xy_batch: jax.Array) -> jax.Array:
        xy_batch = jnp.atleast_2d(xy_batch)
        if xy_batch.shape[0] == 0:
            return jnp.zeros((0,))

        def nn_fn(pt):
            # This reshape is the critical fix for the matrix multiplication error
            pt_reshaped = jnp.reshape(pt, (1, -1))
            return model.raw_output(pt_reshaped).squeeze()

        # Get N and its derivatives
        nn_out, grad_nn = jax.vmap(jax.value_and_grad(nn_fn))(xy_batch)
        nn_x, nn_y = grad_nn[:, 0], grad_nn[:, 1]
        x, y = xy_batch[:, 0], xy_batch[:, 1]

        # Apply the product rule
        u_x = y * nn_out + x * y * nn_x
        u_y = x * nn_out + x * y * nn_y

        # Calculate the PDE residual
        pde_residual = (u_x + u_y) - self.rhs(xy_batch)
        return pde_residual

    # --- Other methods are unchanged ---
    def _single_res(self, model, xy_batch):
        r = self._pointwise_res(model, xy_batch)
        return jnp.mean(r**2)

    def pointwise_residual(self, model, xy):
        return self._pointwise_res(model, xy)

    def residual(self, model, xy):
        return self._single_res(model, xy)


class FirstOrderFreq1010(PDEProblem):
    """
    一阶 PDE:
          u_x + u_y = f(x,y)
          u|_{Γ_in} = 0 ,  Γ_in = {(0,y)} ∪ {(x,0)}
    精确解  u = sin(10π x²) · sin(10π y²)
    """

    domain = (jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))

    # ---------- exact solution ----------
    @staticmethod
    def exact(xy: jax.Array) -> jax.Array:
        xy = jnp.atleast_2d(xy)
        x, y = xy[:, 0], xy[:, 1]
        u = jnp.sin(10 * jnp.pi * x**2) * jnp.sin(10 * jnp.pi * y**2)
        return u[:, None]                            # (N,1)

    # ---------- Dirichlet-0 ansatz ----------
    @staticmethod
    def ansatz(xy: jax.Array, nn_out: jax.Array) -> jax.Array:
        x, y = xy[..., 0], xy[..., 1]
        factor = x * y                               # 保证在 x=0 或 y=0 为 0
        return factor[..., None] * nn_out            # 广播到 (N,1)

    # ---------- RHS  f = u_x + u_y ----------
    @staticmethod
    def rhs(xy: jax.Array) -> jax.Array:
        xy = jnp.atleast_2d(xy)
        x, y = xy[:, 0], xy[:, 1]
        α = β = 10 * jnp.pi
        term_x = 20 * jnp.pi * x * jnp.cos(α * x**2) * jnp.sin(β * y**2)
        term_y = 20 * jnp.pi * y * jnp.sin(α * x**2) * jnp.cos(β * y**2)
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

        # ∇u = (u_x, u_y)s
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

class FirstOrderFreq1216(PDEProblem):
    """
    一阶 PDE:
          u_x + u_y = f(x,y)
          u|_{Γ_in} = 0 ,             Γ_in = {(0,y)} ∪ {(x,0)}
    精确解：sin(12π x²) · sin(16π y²)
    """

    domain = (jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))

    # ---------- exact solution ----------
    @staticmethod
    def exact(xy: jax.Array) -> jax.Array:
        xy = jnp.atleast_1d(xy)
        x, y = xy[..., 0], xy[..., 1]
        return jnp.sin(12 * jnp.pi * x**2) * jnp.sin(16 * jnp.pi * y**2)

    # ---------- Dirichlet-0 ansatz ----------
    @staticmethod
    def ansatz(xy: jax.Array, nn_out: jax.Array) -> jax.Array:
        x, y = xy[..., 0], xy[..., 1]
        factor = x * y                    # 保持与原先一致
        return factor[..., None] * nn_out

    # ---------- RHS  f = u_x + u_y ----------
    @staticmethod
    def rhs(xy: jax.Array) -> jax.Array:
        x, y = xy[..., 0], xy[..., 1]
        α, β = 12 * jnp.pi, 16 * jnp.pi   # 新频率
        term_x = 24 * jnp.pi * x * jnp.cos(α * x**2) * jnp.sin(β * y**2)
        term_y = 32 * jnp.pi * y * jnp.sin(α * x**2) * jnp.cos(β * y**2)
        return term_x + term_y            # shape (...,)

    # ---------- residuals (only 1st-order derivatives) ----------
    def _pointwise_res(self, model: Callable, xy_batch: jax.Array) -> jax.Array:
        xy_batch = jnp.atleast_2d(xy_batch)          # (N,2)
        if xy_batch.shape[0] == 0:
            return jnp.zeros((0,))

        # 前向：只返回 u (标量)
        def u_fn(pt):
            return model(pt).squeeze()               # scalar

        # ∇u = (u_x, u_y)
        grad_u = jax.vmap(jax.grad(u_fn))(xy_batch)  # (N,2)
        grad_sum = grad_u[:, 0] + grad_u[:, 1]       # (N,)

        r = grad_sum - self.rhs(xy_batch)            # (N,)
        return r

    def _single_res(self, model, xy_batch):
        r = self._pointwise_res(model, xy_batch)
        return jnp.mean(r**2)

    # FBPINN 接口
    def pointwise_residual(self, model, xy):
        return self._pointwise_res(model, xy)

    def residual(self, model, xy):
        return self._single_res(model, xy)


class FirstOrderFreq34(PDEProblem):
    """
    一阶 PDE:
          u_x + u_y = f(x,y)
          u|_{∂Ω} = 0           on Ω = [0,1]²
    Exact solution:
          u(x,y) = sin(3π x²) • sin(4π y²)
    """

    domain = (jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))

    # ---------- exact solution ----------
    @staticmethod
    def exact(xy: jax.Array) -> jax.Array:
        xy = jnp.atleast_1d(xy)
        x, y = xy[..., 0], xy[..., 1]
        return jnp.sin(3 * jnp.pi * x**2) * jnp.sin(4 * jnp.pi * y**2)

    # ---------- Dirichlet-0 ansatz ----------
    @staticmethod
    def ansatz(xy: jax.Array, nn_out: jax.Array) -> jax.Array:
        x, y = xy[..., 0], xy[..., 1]
        factor = x * (1 - x) * y * (1 - y)          # 保证 u=0 on ∂Ω
        return factor[..., None] * nn_out

    # ---------- RHS  f = u_x + u_y ----------
    @staticmethod
    def rhs(xy: jax.Array) -> jax.Array:
        x, y = xy[..., 0], xy[..., 1]

        α, β = 3 * jnp.pi, 4 * jnp.pi               # 新频率
        # u_x = 2α x cos(α x²) sin(β y²)
        term_x = 2 * α * x * jnp.cos(α * x**2) * jnp.sin(β * y**2)
        # u_y = 2β y sin(α x²) cos(β y²)
        term_y = 2 * β * y * jnp.sin(α * x**2) * jnp.cos(β * y**2)

        return term_x + term_y                      # shape (...,)

    # ---------- residuals (only 1st-order derivatives) ----------
    def _pointwise_res(self, model: Callable, xy_batch: jax.Array) -> jax.Array:
        xy_batch = jnp.atleast_2d(xy_batch)         # (N,2)
        if xy_batch.shape[0] == 0:
            return jnp.zeros((0,))

        def u_fn(pt):
            return model(pt).squeeze()              # scalar

        grad_u   = jax.vmap(jax.grad(u_fn))(xy_batch)
        grad_sum = grad_u[:, 0] + grad_u[:, 1]

        r = grad_sum - self.rhs(xy_batch)           # (N,)
        return r

    # ---------- helper wrappers ----------
    def _single_res(self, model, xy_batch):
        r = self._pointwise_res(model, xy_batch)
        return jnp.mean(r**2)

    def pointwise_residual(self, model, xy):
        return self._pointwise_res(model, xy)

    def residual(self, model, xy):
        return self._single_res(model, xy)
    
class CosRHSFreq15(PDEProblem):
    """
    PDE:
        ∂u/∂x₁ + ∂u/∂x₂ = cos(ω x₁) + cos(ω x₂),       ω = 15
        u(0, x₂) = (1/ω)·sin(ω x₂)                    on Ω = [-2π, 2π]²

    Exact solution (Eq. 23 in the paper):
        u(x₁, x₂) = (1/ω)·sin(ω x₁) + (1/ω)·sin(ω x₂)
    """

    ω: float = 15.0                       # frequency parameter
    L: float = 2 * jnp.pi                 # half-length of the domain

    # ------------- domain -------------
    domain = (jnp.array([-L, -L]), jnp.array([L, L]))   # shape (2,)

    # ---------- exact solution ----------
    @staticmethod
    def exact(xy: jax.Array) -> jax.Array:
        xy = jnp.atleast_1d(xy)
        x1, x2 = xy[..., 0], xy[..., 1]
        ω = CosRHSFreq15.ω
        return (jnp.sin(ω * x1) + jnp.sin(ω * x2)) / ω    # shape (...)

    # ---------- Dirichlet ansatz (Eq. 24) ----------
    @staticmethod
    def ansatz(xy: jax.Array, nn_out: jax.Array) -> jax.Array:
        x1, x2 = xy[..., 0], xy[..., 1]
        ω = CosRHSFreq15.ω
        base   = (1.0 / ω) * jnp.sin(ω * x2)             # satisfies u(0,x2)
        factor = jnp.tanh(ω * x1)                        # vanishes at x1 = 0
        return base[..., None] + factor[..., None] * nn_out

    # ---------- RHS  f = cos(ωx₁)+cos(ωx₂) ----------
    @staticmethod
    def rhs(xy: jax.Array) -> jax.Array:
        x1, x2 = xy[..., 0], xy[..., 1]
        ω = CosRHSFreq15.ω
        return jnp.cos(ω * x1) + jnp.cos(ω * x2)          # shape (...)

    # ---------- residuals (first-order only) ----------
    def _pointwise_res(self, model: Callable, xy_batch: jax.Array) -> jax.Array:
        xy_batch = jnp.atleast_2d(xy_batch)               # (N,2)
        if xy_batch.shape[0] == 0:
            return jnp.zeros((0,))

        # forward pass: scalar u
        def u_fn(pt):
            return model(pt).squeeze()

        # ∇u = (u_x1, u_x2)
        grad_u = jax.vmap(jax.grad(u_fn))(xy_batch)       # (N,2)
        grad_sum = grad_u[:, 0] + grad_u[:, 1]            # (N,)

        r = grad_sum - self.rhs(xy_batch)                 # (N,)
        return r

    def _single_res(self, model, xy_batch):
        r = self._pointwise_res(model, xy_batch)
        return jnp.mean(r ** 2)

    # FBPINN 接口 --------------------------------------------------
    def pointwise_residual(self, model, xy):
        return self._pointwise_res(model, xy)

    def residual(self, model, xy):
        return self._single_res(model, xy)
    
import jax
import jax.numpy as jnp

class FirstOrderFreq68:
    """
    Solves the first-order PDE:
        ∂u/∂x + ∂u/∂y = f(x,y)
    with the boundary condition u=0 on x=0 and y=0.
    """
    domain = (jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))

    @staticmethod
    def exact(xy: jax.Array) -> jax.Array:
        """The exact solution to the PDE problem."""
        x, y = xy[..., 0], xy[..., 1]
        return jnp.sin(6 * jnp.pi * x**2) * jnp.sin(8 * jnp.pi * y**2)

    @staticmethod
    def ansatz(xy: jax.Array, nn_out: jax.Array) -> jax.Array:
        """Builds the solution by enforcing zero Dirichlet boundary conditions."""
        x, y = xy[..., 0], xy[..., 1]
        return (x * y)[..., None] * nn_out

    @staticmethod
    def rhs(xy: jax.Array) -> jax.Array:
        """The right-hand side of the PDE, f(x,y)."""
        x, y = xy[..., 0], xy[..., 1]
        alpha, beta = 6 * jnp.pi, 8 * jnp.pi
        term_x = 12 * jnp.pi * x * jnp.cos(alpha * x**2) * jnp.sin(beta * y**2)
        term_y = 16 * jnp.pi * y * jnp.sin(alpha * x**2) * jnp.cos(beta * y**2)
        return term_x + term_y

    def _pointwise_res(self, model: Callable, xy_batch: jax.Array) -> jax.Array:
        """Calculates the residual of the PDE at each point."""
        xy_batch = jnp.atleast_2d(xy_batch)

        def u_fn(pt):
            # The model's output for a single point
            return jnp.ravel(model(pt[None, :]))[0]

        # Compute u and its gradient (u_x, u_y) for the batch
        u_val, grad_u = jax.vmap(jax.value_and_grad(u_fn))(xy_batch)
        u_x, u_y = grad_u[:, 0], grad_u[:, 1]
        
        # PDE Residual: (u_x + u_y) - f = 0
        return (u_x + u_y) - self.rhs(xy_batch)

    def residual(self, model: Callable, xy: jax.Array) -> jax.Array:
        """Calculates the mean squared error loss from the residuals."""
        r = self._pointwise_res(model, xy)
        return jnp.mean(r**2)


class Wave1DHighFreq(PDEProblem):
    """
    1‑D wave equation on Ω = [0,1]×[0,1]:

        u_tt − 4 u_xx = 0 ,  (c = 2)
        u(0,t) = u(1,t) = 0          (Dirichlet)
        u(x,0) = sin(πx) + 0.5·sin(4πx)
        u_t(x,0) = 0

    Exact solution:
        u(x,t) = sin(πx)·cos(2πt) + 0.5·sin(4πx)·cos(8πt)
    """

    # ------------------------------------------------------------------
    # domain in (x, t)
    domain = (
        jnp.array([0.0, 0.0]),   # lower bounds
        jnp.array([1.0, 1.0]),   # upper bounds
    )

    # ------------------------------------------------------------------
    # ▶ exact solution --------------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def exact(xt: jax.Array) -> jnp.ndarray:  # (...,)
        xt = jnp.atleast_1d(xt)
        x, t = xt[..., 0], xt[..., 1]
        term1 = jnp.sin(jnp.pi * x) * jnp.cos(2 * jnp.pi * t)
        term2 = 0.5 * jnp.sin(4 * jnp.pi * x) * jnp.cos(8 * jnp.pi * t)
        return term1 + term2

    # ------------------------------------------------------------------
    # ▶ Dirichlet + IC hard‑constrained ansatz -------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def u0(x: jax.Array) -> jnp.ndarray:
        """Initial displacement u(x,0)."""
        return jnp.sin(jnp.pi * x) + 0.5 * jnp.sin(4 * jnp.pi * x)

    @staticmethod
    def ansatz(xt: jax.Array, nn_out: jax.Array) -> jnp.ndarray:
        """Trial solution that satisfies BC & IC exactly.

        û(x,t;θ) = u0(x) + x(1−x)·t² · NN(x,t;θ)
        """
        x, t = xt[..., 0], xt[..., 1]
        mask = x * (1.0 - x) * (t ** 2)
        return Wave1DHighFreq.u0(x)[..., None] + mask[..., None] * nn_out

    # ------------------------------------------------------------------
    # ▶ RHS  f(x,t) = 0 ------------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def rhs(_: jax.Array) -> jnp.ndarray:
        return 0.0

    # ------------------------------------------------------------------
    # ▶ point‑wise residual --------------------------------------------
    # ------------------------------------------------------------------
    def _pointwise_res(self, model: Callable, xt_batch: jax.Array) -> jnp.ndarray:
        """Compute r(x,t) = u_tt − 4 u_xx for a batch of points."""
        xt_batch = jnp.atleast_2d(xt_batch)      # (N, 2)
        if xt_batch.shape[0] == 0:
            return jnp.zeros((0,))

        # scalar forward
        def u_fn(pt):
            return model(pt).squeeze()

        # second derivatives via nested grad (avoids computing u_xt)
        def u_xx(pt):
            return jax.grad(lambda z: jax.grad(u_fn)(z)[0])(pt)

        def u_tt(pt):
            return jax.grad(lambda z: jax.grad(u_fn)(z)[1])(pt)

        u_xx_arr = jax.vmap(u_xx)(xt_batch)      # (N,)
        u_tt_arr = jax.vmap(u_tt)(xt_batch)      # (N,)

        return u_tt_arr - 4.0 * u_xx_arr         # (N,)

    # ------------------------------------------------------------------
    # ▶ mean‑square residual -------------------------------------------
    # ------------------------------------------------------------------
    def _single_res(self, model: Callable, xt_batch: jax.Array) -> jnp.ndarray:
        r = self._pointwise_res(model, xt_batch)
        return jnp.mean(r ** 2)

    # ------------------------------------------------------------------
    # ▶ Public interfaces (trainer hooks) ------------------------------
    # ------------------------------------------------------------------
    def pointwise_residual(self, model: Callable, xt: jax.Array) -> jnp.ndarray:
        return self._pointwise_res(model, xt)

    def residual(self, model: Callable, xt: jax.Array) -> jnp.ndarray:
        return self._single_res(model, xt)

class FirstOrderConvect105(PDEProblem):
    r"""
    一阶对流 PDE:
          u_x + 0.5 u_y = f(x,y)
          u|_{Γ_in}  = 0 ,           Γ_in = {(0,y)} ∪ {(x,0)}
    选取精确解   u(x,y) = sin(6π x²) · sin(8π y²)

    - 对流速度向量 **v = (1.0, 0.5)**
    - 计算残差   v·∇u − f
    """

    # ---------- 计算域 ----------
    domain = (jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))

    # ---------- 精确解 ----------
    @staticmethod
    def exact(xy: jax.Array) -> jax.Array:
        xy = jnp.atleast_1d(xy)
        x, y = xy[..., 0], xy[..., 1]
        return jnp.sin(6 * jnp.pi * x**2) * jnp.sin(8 * jnp.pi * y**2)

    # ---------- Dirichlet-0 ansatz ----------
    @staticmethod
    def ansatz(xy: jax.Array, nn_out: jax.Array) -> jax.Array:
        x, y = xy[..., 0], xy[..., 1]
        factor = x * y                      # 保证在 x=0 或 y=0 处为 0
        return factor[..., None] * nn_out   # 保留 NN 输出的形状

    # ---------- RHS: f = u_x + 0.5 u_y ----------
    @staticmethod
    def rhs(xy: jax.Array) -> jax.Array:
        x, y = xy[..., 0], xy[..., 1]
        α, β = 6 * jnp.pi, 8 * jnp.pi

        u_x = 2 * α * x * jnp.cos(α * x**2) * jnp.sin(β * y**2)        # 12π x cos… sin…
        u_y = 2 * β * y * jnp.sin(α * x**2) * jnp.cos(β * y**2)        # 16π y sin… cos…
        return u_x + 0.5 * u_y                                          # shape (...,)

    # ---------- pointwise residual ----------
    def _pointwise_res(self, model: Callable, xy_batch: jax.Array) -> jax.Array:
        xy_batch = jnp.atleast_2d(xy_batch)          # (N, 2)
        if xy_batch.shape[0] == 0:
            return jnp.zeros((0,))

        # 前向: 仅返回标量 u
        def u_fn(pt):
            return model(pt).squeeze()               # scalar

        # ∇u = (u_x, u_y)
        grad_u = jax.vmap(jax.grad(u_fn))(xy_batch)  # (N, 2)
        # v·∇u  = 1.0 * u_x + 0.5 * u_y
        conv_term = grad_u[:, 0] + 0.5 * grad_u[:, 1]

        r = conv_term - self.rhs(xy_batch)           # (N,)
        return r

    # ---------- scalar residual (mean-squared) ----------
    def _single_res(self, model, xy_batch):
        r = self._pointwise_res(model, xy_batch)
        return jnp.mean(r**2)

    # ---------- FBPINN 接口 ----------
    def pointwise_residual(self, model, xy):
        return self._pointwise_res(model, xy)

    def residual(self, model, xy):
        return self._single_res(model, xy)
    


class LocalHighFreqPoisson1D(PDEProblem):
    """
    Poisson benchmark with one global low-frequency mode and one
    narrow-band high-frequency Gaussian packet.

    Equation            :  u''(x) + f(x) = 0 ,  x ∈ [0, 1]
    Exact solution      :  u(x) = sin(k_low * x)
                           + exp(-γ (x - xc)**2) * sin(k_high * x)
    Boundary constraint :  u(0) = u(1) = 0  (hard-encoded ansatz)
    """

    # ---- domain & parameters ------------------------------------------------
    domain = (jnp.array([0.0]), jnp.array([1.0]))   # Dirichlet at both ends

    k_low: float   = 2.0 * jnp.pi     # 2 π → 1 global oscillation
    k_high: float  = 40.0 * jnp.pi    # 40 π → 20 local oscillations
    gamma: float   = 100.0            # envelope strength exp(-γ⋅(x-xc)²)
    xc: float      = 0.7              # centre of the high-freq packet

    # ---- exact solution & forcing term -------------------------------------
    @staticmethod
    def exact(x: jnp.ndarray) -> jnp.ndarray:
        """
        Analytical solution u(x) for any x ∈ ℝ or array-like x.
        """
        cls = LocalHighFreqPoisson1D          # shorthand
        low  = jnp.sin(cls.k_low  * x)
        high = jnp.exp(-cls.gamma * (x - cls.xc) ** 2) * jnp.sin(cls.k_high * x)
        return low + high

    def _f(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forcing term f(x) such that u''(x) + f(x) = 0 .
        We get it by auto-diff:  f(x) = -u''(x).
        """
        # flatten because jax.grad operates on scalar function
        x = jnp.ravel(x)

        def u_scalar(xx):
            return jnp.sum(self.exact(xx))     # → scalar

        u_xx = jax.vmap(jax.grad(jax.grad(u_scalar)))(x)
        return u_xx                           # f(x) = -u''(x)

    # ---- hard Dirichlet boundary via ansatz ---------------------------------
    @staticmethod
    def ansatz(x: jnp.ndarray, nn_out: jnp.ndarray) -> jnp.ndarray:
        """
        Embeds Dirichlet BCs  u(0)=u(1)=0  into the network output.
        """
        left, right = LocalHighFreqPoisson1D.domain
        return (x - left.item()) * (right.item() - x) * nn_out

    # ---- residual -----------------------------------------------------------
    def _single_res(self, model, x: jnp.ndarray) -> jnp.ndarray:
        """
        PDE residual (mean-squared) on a batch x for one subnet / shard.
        """
        x = jnp.ravel(x)

        def u_scalar(xx):
            return jnp.sum(model(xx))          # model already wrapped by ansatz

        u_xx = jax.vmap(jax.grad(jax.grad(u_scalar)))(x)
        res  = u_xx + self._f(x)
        return jnp.mean(res ** 2)

    def residual(self, model, x: Union[jnp.ndarray, Sequence[jnp.ndarray]]
                 ) -> jnp.ndarray:
        """
        Supports a list / tuple of batches (e.g. per-subdomain) or a single batch.
        """
        if isinstance(x, (list, tuple)):
            losses = [self._single_res(model, xi) for xi in x]
            return jnp.sum(jnp.stack(losses))
        else:
            return self._single_res(model, x)


import numpy as np
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple

import jax, jax.numpy as jnp, numpy as np
from typing import Tuple

class ViscousBurgersFBPINN(PDEProblem):
    # ------------------------------------------------------------
    # 0. Basic constants and domain (xt[...,0]=x , xt[...,1]=t)
    # ------------------------------------------------------------
    nu: float = 0.01 / jnp.pi
    domain: Tuple[jnp.ndarray, jnp.ndarray] = (
        jnp.array([-1.0, 0.0]),   # x_min , t_min
        jnp.array([1.0, 1.0]),    # x_max , t_max
    )

    # ------------------------------------------------------------
    # 1. Precompute Gauss-Hermite nodes/weights (constant, put on CPU)
    # ------------------------------------------------------------
    _QN = 60
    _QX_np, _QW_np = None, None  # Stay resident after filling once

    @staticmethod
    def _init_quadrature():
        if ViscousBurgersFBPINN._QX_np is None:
            n = ViscousBurgersFBPINN._QN
            i = jnp.arange(1.0, n)
            a = jnp.sqrt(i / 2.0)
            J = jnp.diag(a, -1) + jnp.diag(a, 1)
            x, v = jnp.linalg.eigh(J)
            w = jnp.pi**(-0.25) * jnp.exp(x**2 / 2.0) * v[0] ** 2
            ViscousBurgersFBPINN._QX_np = x.astype(jnp.float32)
            ViscousBurgersFBPINN._QW_np = w.astype(jnp.float32)

    # ------------------------------------------------------------
    # 2. Exact Solution (point-wise) (Can be JIT / vmap)
    # ------------------------------------------------------------
    @staticmethod
    def _exact_point(x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Single point exact solution u_exact(x,t) → scalar, supports broadcasting.
        """
        nu = ViscousBurgersFBPINN.nu
        qx = ViscousBurgersFBPINN._QX_np
        qw = ViscousBurgersFBPINN._QW_np

        def nonzero_time(xi, ti):
            c = 2.0 * jnp.sqrt(nu * ti)
            top = 0.0
            bot = 0.0

            # Correctly calculate top and bot
            for qi in range(0, ViscousBurgersFBPINN._QN):
                arg = jnp.pi * (xi - c * qx[qi])
                e = jnp.exp(-jnp.cos(arg) / (2.0 * jnp.pi * nu))

                # Calculate top and bot
                top += -qw[qi] * c * jnp.sin(arg) * e
                bot += qw[qi] * c * e

            # Avoid division by zero by ensuring bot is not zero
            bot = jnp.where(bot == 0, 1e-8, bot)  # Ensure bot is not zero
            return top / bot

        # If t == 0, return initial condition
        return jnp.where(t == 0.0,
                         -jnp.sin(jnp.pi * x),  # Fix here xi -> x
                         nonzero_time(x, t))

    # ------------------------------------------------------------
    # 3. Public exact solution — 任意形 (…,2) → (…,1)
    # ------------------------------------------------------------
    @staticmethod
    def exact(xt: jnp.ndarray) -> jnp.ndarray:
        """
        - Input: shape (...,2) or (2,)  (x,t)
        - Output: (...,1)
        """
        ViscousBurgersFBPINN._init_quadrature()  # Initialize quadrature (only once)
        xt = jnp.asarray(xt)
        x, t = xt[..., 0], xt[..., 1]  # broadcast-safe
        u = ViscousBurgersFBPINN._exact_point(x, t)
        return u[..., None]  # (...,1)

    # ------------------------------------------------------------
    # 4. Ansatz — IC + Dirichlet BC
    # ------------------------------------------------------------
    @staticmethod
    def ansatz(xt: jax.Array, nn_out: jax.Array) -> jax.Array:
        x, t = xt[..., 0:1], xt[..., 1:2]
        env = jnp.tanh(x + 1) * jnp.tanh(x - 1) * jnp.tanh(t)
        return -jnp.sin(jnp.pi * x) + env * nn_out

    # ------------------------------------------------------------
    # 5. Pointwise residual  R = u_t + u u_x − ν u_xx
    # ------------------------------------------------------------
    def _pointwise_res(self, model: Callable, xt_batch: jax.Array) -> jax.Array:
        xt_batch = jnp.atleast_2d(xt_batch)  # (N,2)

        def u(pt):
            return model(pt).squeeze()  # scalar

        grad_fn, hess_fn = jax.grad(u), jax.hessian(u)

        grad = jax.vmap(grad_fn)(xt_batch)  # (N,2)
        u_x, u_t = grad[:, 0], grad[:, 1]

        hess = jax.vmap(hess_fn)(xt_batch)  # (N,2,2)
        u_xx = hess[:, 0, 0]

        u_val = jax.vmap(u)(xt_batch)  # (N,)
        res = u_t + u_val * u_x - self.nu * u_xx
        return res[:, None]  # (N,1)

    # ------------------------------------------------------------
    # 6. FBPINN required interface
    # ------------------------------------------------------------
    def pointwise_residual(self, model, xt):
        return self._pointwise_res(model, xt)

    def residual(self, model, xt):
        return jnp.mean(self._pointwise_res(model, xt) ** 2)