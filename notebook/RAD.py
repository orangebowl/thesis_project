from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import matplotlib.pyplot as plt
from typing import Callable, Sequence

class PDEProblem:
    domain = (None, None)
    def residual(self, model, x):
        """用于训练的 PDE 残差(均值)。"""
        raise NotImplementedError
    def exact(self, x):
        """若有真解可用于计算误差。"""
        raise NotImplementedError
    def ansatz(self, x, nn_out):
        """处理边界条件。"""
        raise NotImplementedError

class Poisson2D_freq(PDEProblem):
    """
    Poisson: -Δu = f(x,y),   u|∂Ω=0  on [0,1]^2
    真解(测试用):  u(x,y)= sin(2π x²)*sin(2π y²)
    """
    domain = (jnp.array([0.,0.]), jnp.array([1.,1.]))

    @staticmethod
    def exact(xy):
        x, y = xy[...,0], xy[...,1]
        return jnp.sin(2*jnp.pi*x**2)*jnp.sin(2*jnp.pi*y**2)

    @staticmethod
    def ansatz(xy, nn_out):
        """强制0边界: x(1-x)*y(1-y)*nn_out"""
        x, y = xy[...,0], xy[...,1]
        return (x*(1-x)*y*(1-y))[...,None]*nn_out

    @staticmethod
    def rhs(xy):
        x, y = xy[...,0], xy[...,1]
        sin_x2 = jnp.sin(2*jnp.pi*x**2)
        sin_y2 = jnp.sin(2*jnp.pi*y**2)
        cos_x2 = jnp.cos(2*jnp.pi*x**2)
        cos_y2 = jnp.cos(2*jnp.pi*y**2)
        term1 = -4*jnp.pi*(sin_y2*cos_x2 + sin_x2*cos_y2)
        term2 = 16*(jnp.pi**2)*(x**2+y**2)*sin_x2*sin_y2
        return term1 + term2

    def _single_res(self, model, xy):
        if xy.shape[0]==0:
            return 0.0
        def u_fn(pt):
            out = model(pt)
            return out.squeeze()
        # 二阶导 => Laplacian
        hessian_fn = jax.jacfwd(jax.jacrev(u_fn))
        hessians   = jax.vmap(hessian_fn)(xy)
        laplacians = jnp.trace(hessians, axis1=-2, axis2=-1)
        f_vals = self.rhs(xy)
        return jnp.mean(( -laplacians - f_vals )**2)

    def residual(self, model, xy):
        return self._single_res(model, xy)

# pointwise_residual for RAD

def pointwise_residual(pde: PDEProblem, model: Callable, xy: jnp.ndarray) -> jnp.ndarray:
    """
    返回每个点的 PDE 残差(绝对值), shape=(N,).
    Poisson: eps = | -(laplacian(u)) - f |.
    """
    def u_fn(pt):
        out = model(pt)
        return out.squeeze()
    hessian_fn = jax.jacfwd(jax.jacrev(u_fn))
    hessians   = jax.vmap(hessian_fn)(xy)
    laplacians = jnp.trace(hessians, axis1=-2, axis2=-1)
    f_vals = pde.rhs(xy)
    return jnp.abs(-laplacians - f_vals)


def generate_subdomains(domain, n_sub_per_dim, overlap):
    """
    在 domain=[(0,0),(1,1)] 上, 每个维度 n_sub_per_dim 个子域(带 overlap).
    返回 subdomains=[(left,right),...], 其中 left,right形如(2,).
    """
    if isinstance(n_sub_per_dim,int):
        n_sub_per_dim= [n_sub_per_dim]*len(domain[0])
    dim = len(domain[0])
    grid_axes=[]
    step_sizes=[]
    for i in range(dim):
        a, b= domain[0][i], domain[1][i]
        n= n_sub_per_dim[i]
        total_len= b-a
        step= total_len/(n-1)
        centers= jnp.linspace(a,b,n)
        grid_axes.append(centers)
        step_sizes.append(step)
    mesh= jnp.meshgrid(*grid_axes, indexing="ij")
    cpts= jnp.stack([m.reshape(-1) for m in mesh], axis=-1)
    subdomains=[]
    for center in cpts:
        width= jnp.array(step_sizes)/2 + overlap/2
        left = center - width
        right= center + width
        subdomains.append((left, right))
    return subdomains

def generate_whole_domain_collocation(domain, n_pts):
    """在 [0,1]^2 上生成 n_pts×n_pts 的网格"""
    (x_lo,y_lo),(x_hi,y_hi)= domain
    xs= jnp.linspace(x_lo, x_hi, n_pts)
    ys= jnp.linspace(y_lo, y_hi, n_pts)
    XX,YY= jnp.meshgrid(xs, ys, indexing="ij")
    return jnp.column_stack([XX.ravel(), YY.ravel()])


import jax
import jax.numpy as jnp
Pi = jnp.pi

def my_precise_window_func(xmins_all, xmaxs_all, wmins_all, wmaxs_all, x, tol=1e-12):
    """
    2D Cos² window (FBPINN-style), 先不详细展开
    """
    xmins_all= jnp.asarray(xmins_all)
    xmaxs_all= jnp.asarray(xmaxs_all)
    x = jnp.atleast_2d(x)
    n_sub= xmins_all.shape[0]

    def single_window(i_sub, X):
        xmin= xmins_all[i_sub]
        xmax= xmaxs_all[i_sub]
        mu= (xmin+xmax)/2
        sd= (xmax-xmin)/2 + tol
        r= (X-mu)/sd
        core= 0.25*(1.+ jnp.cos(Pi*r))**2
        mask= jnp.where(jnp.abs(r)<=1., core, 0.)
        return jnp.prod(mask, axis=-1)

    w_raw= jax.vmap(lambda i: single_window(i,x))(jnp.arange(n_sub)).T
    denom= jnp.maximum(w_raw.sum(axis=1,keepdims=True), 1e-12)
    return w_raw/denom


# FBPINN /import from model.py later
# Dense layer
# Dense layer
# Dense layer
class Dense(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    activation: Callable | None = eqx.static_field()

    def __init__(
        self,
        key: jax.Array,
        in_features: int,
        out_features: int,
        activation: Callable | None = jax.nn.relu,
    ):
        w_key, b_key = jax.random.split(key)
        limit = jnp.sqrt(1.0 / in_features)
        self.weight = jax.random.uniform(
            w_key, (out_features, in_features), minval=-limit, maxval=limit
        )
        self.bias = jax.random.uniform(
            b_key, (out_features,), minval=-limit, maxval=limit
        )
        self.activation = activation

    def __call__(self, x: jax.Array) -> jax.Array:
        y = x @ self.weight.T + self.bias
        return self.activation(y) if self.activation is not None else y


class FCN(eqx.Module):
    """MLP"""
    layers: tuple[Dense, ...]

    def __init__(
        self,
        key: jax.Array,
        in_size: int,
        out_size: int,
        hidden_sizes: Sequence[int] | int = 64,
        activation: Callable = jax.nn.tanh,
        final_activation: Callable | None = None,
    ):
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        keys = jax.random.split(key, len(hidden_sizes) + 1)
        sizes = [in_size, *hidden_sizes, out_size]

        self.layers = tuple(
            Dense(
                k,
                in_features=sizes[i],
                out_features=sizes[i + 1],
                activation=activation if i < len(hidden_sizes) else final_activation,
            )
            for i, k in enumerate(keys)
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.atleast_2d(x)
        for dense in self.layers:
            x = dense(x)
        return x

class FBPINN(eqx.Module):
    subnets: tuple
    ansatz: callable= eqx.static_field()
    xmins_all: jax.Array= eqx.static_field()
    xmaxs_all: jax.Array= eqx.static_field()
    wmins_all_fixed: jax.Array= eqx.static_field()
    wmaxs_all_fixed: jax.Array= eqx.static_field()
    num_subdomains: int= eqx.static_field()
    xdim: int= eqx.static_field()
    model_out_size: int= eqx.static_field()

    def __init__(self, key, subdomains, ansatz, mlp_config, fixed_transition):
        self.ansatz= ansatz
        self.xdim= mlp_config["in_size"]
        self.model_out_size= mlp_config["out_size"]

        if not subdomains:
            self.num_subdomains=0
            self.subnets= tuple()
            pshape= (0,self.xdim)
            self.xmins_all= jnp.empty(pshape)
            self.xmaxs_all= jnp.empty(pshape)
            self.wmins_all_fixed= jnp.empty(pshape)
            self.wmaxs_all_fixed= jnp.empty(pshape)
        else:
            self.num_subdomains= len(subdomains)
            s_mins= [s[0] for s in subdomains]
            s_maxs= [s[1] for s in subdomains]
            self.xmins_all= jnp.stack(s_mins)
            self.xmaxs_all= jnp.stack(s_maxs)
            self.wmins_all_fixed= jnp.full((self.num_subdomains,self.xdim), fixed_transition)
            self.wmaxs_all_fixed= jnp.full((self.num_subdomains,self.xdim), fixed_transition)

            keys= jax.random.split(key, self.num_subdomains)
            self.subnets= tuple(
                FCN(k, in_size=2, out_size=1, hidden_sizes=[64, 64], activation=jax.nn.tanh)  
                for k in keys
            )

    def _normalize_x(self, i_sub, x):
        left= self.xmins_all[i_sub]
        right= self.xmaxs_all[i_sub]
        center= (left+right)/2.
        scale= (right-left)/2.
        return (x-center)/ jnp.maximum(scale, 1e-9)

    def total_solution(self, x):
        """
        x shape=(N,2).
        We do subdomain-wise MLP + weight, then sum.
        """
        if self.num_subdomains==0:
            return jnp.zeros_like(x[...,0:1])

        w_raw= my_precise_window_func(self.xmins_all, self.xmaxs_all,
                                      self.wmins_all_fixed, self.wmaxs_all_fixed,
                                      x, tol=1e-8)
        out_list=[]
        for i_sub in range(self.num_subdomains):
            xnorm= self._normalize_x(i_sub, x)
            raw_i= self.subnets[i_sub](xnorm)
            w_i= w_raw[:, i_sub]
            out_i= raw_i*w_i[:,None] if raw_i.ndim==2 else raw_i*w_i
            out_list.append(out_i)
        sum_out= jnp.sum(jnp.stack(out_list,axis=0), axis=0)
        return self.ansatz(x, sum_out)

    def __call__(self, x):
        return self.total_solution(x)

# RAD sample
def sample_by_RAD(
    key: jax.Array,
    pde: PDEProblem,
    model: Callable,
    num_points: int,
    k: float=1.0,
    c: float=1.0,
    domain=(jnp.array([0.,0.]), jnp.array([1.,1.])),
    pool_size: int=2000
)-> jnp.ndarray:
    """
    与“固定池”不同的是，这里每次都在 domain 内随机生成pool_size个点(不会复用),
    然后根据残差分布离散采样 num_points.???????????????????????????????
    """
    key_x, key_y= jax.random.split(key)
    (xlo,ylo),(xhi,yhi)= domain
    xs= jax.random.uniform(key_x,(pool_size,),minval=xlo,maxval=xhi)
    ys= jax.random.uniform(key_y,(pool_size,),minval=ylo,maxval=yhi)
    xy_pool= jnp.column_stack([xs, ys])  # shape=(pool_size,2)

    eps_vals= pointwise_residual(pde, model, xy_pool)
    epsk_vals= eps_vals**k
    mean_epsk= jnp.maximum(jnp.mean(epsk_vals),1e-12)
    p_raw= epsk_vals/mean_epsk + c
    p_sum= jnp.maximum(jnp.sum(p_raw), 1e-12)
    p= p_raw/p_sum

    def draw_one(carry,_):
        subk, prob= carry
        subk, use_key= jax.random.split(subk)
        idx= jax.random.categorical(use_key, jnp.log(prob))
        return (subk, prob), xy_pool[idx]

    (_,_), draws= jax.lax.scan(draw_one,(key,p),None,length=num_points)
    return draws


# Training loop for RAD
def rad_training_loop(
    key: jax.Array,
    pde: PDEProblem,
    fb_model: FBPINN,
    optimizer,
    test_pts: jnp.ndarray,
    steps_per_stage: int=1000,
    rad_stages: int=5,
    init_num_pts: int=2000,
    rad_num_pts: int=2000,
    rad_pool_size: int=5000,
    k: float=1.0,
    c: float=1.0
):
    """
    1) 初始随机点 init_num_pts => 训练
    2) 每阶段结束 => 调用 sample_by_RAD (不复用任何池) => 替换 colloc_pts
    3) 记录 train_loss & test_l1
    """
    (xlo,ylo),(xhi,yhi)= pde.domain
    # A) 先随机初始化
    key_x, key_y, key= jax.random.split(key,3)
    xs= jax.random.uniform(key_x,(init_num_pts,), xlo,xhi)
    ys= jax.random.uniform(key_y,(init_num_pts,), ylo,yhi)
    colloc_pts= jnp.column_stack([xs, ys])

    # B) 优化器
    opt_state= optimizer.init(eqx.filter(fb_model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_s, x_batch):
        loss_val, grads= eqx.filter_value_and_grad(lambda m: pde.residual(m, x_batch))(model)
        updates, opt_s_new= optimizer.update(grads, opt_s, model)
        model_new= eqx.apply_updates(model, updates)
        return model_new, opt_s_new, loss_val

    def eval_test_l1(model, x_test):
        pred= model(x_test).reshape(-1)
        true= pde.exact(x_test).reshape(-1)
        return jnp.mean(jnp.abs(pred- true))

    step_count=0
    train_loss_list=[]
    test_l1_list=[]
    step_list=[]
    total_steps= steps_per_stage*rad_stages
    log_interval=500

    for stage_i in range(rad_stages):
        # 1) train steps_per_stage
        for _ in range(steps_per_stage):
            fb_model, opt_state, loss_val= train_step(fb_model, opt_state, colloc_pts)
            step_count+=1
            if (step_count%log_interval==0) or (step_count==total_steps):
                train_loss= loss_val.item()
                l1_err= eval_test_l1(fb_model, test_pts).item()
                print(f"[Stage={stage_i}] step={step_count}, TrainLoss={train_loss:.3e}, TestL1={l1_err:.3e}")
                train_loss_list.append(train_loss)
                test_l1_list.append(l1_err)
                step_list.append(step_count)

        # 2) 如果还没到最后，就重采样 => sample_by_RAD
        if stage_i< rad_stages-1:
            rad_key, key= jax.random.split(key)
            new_pts= sample_by_RAD(
                rad_key, pde, fb_model,
                num_points= rad_num_pts,
                k=k, c=c,
                domain= pde.domain,
                pool_size= rad_pool_size
            )
            colloc_pts= new_pts  # 全替换
            # 可视化
            plt.figure()
            plt.scatter(new_pts[:,0], new_pts[:,1], s=6, c="red", alpha=0.7)
            plt.title(f"New Collocation after Stage={stage_i+1}")
            plt.xlim([xlo,xhi])
            plt.ylim([ylo,yhi])
            plt.grid(True, ls=":")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()

    return fb_model, (step_list, train_loss_list, test_l1_list)



def rad_training_loop(
    key: jax.Array,
    pde: PDEProblem,
    fb_model: FBPINN,
    optimizer,
    test_pts: jnp.ndarray,   # 用于测 L1
    steps_per_stage: int=1000,
    rad_stages: int=5,
    init_num_pts: int=2000,
    rad_num_pts: int=2000,
    rad_pool_size: int=5000,
    k: float=1.0,
    c: float=1.0
):
    """
    1) 初始随机点 init_num_pts => 训练
    2) 每段结束后 => RAD 重采样 rad_num_pts 点 => 可视化
    3) 同时记录 train_loss & test_l1
    """
    (xlo,ylo),(xhi,yhi)= pde.domain
    # (A) 先随机初始化 collocation points
    key_x, key_y, key= jax.random.split(key,3)
    xs= jax.random.uniform(key_x, (init_num_pts,), minval=xlo, maxval=xhi)
    ys= jax.random.uniform(key_y, (init_num_pts,), minval=ylo, maxval=yhi)
    colloc_pts= jnp.column_stack([xs,ys])

    # (B) 优化器
    opt_state= optimizer.init(eqx.filter(fb_model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_s, x_batch):
        loss_val, grads= eqx.filter_value_and_grad(lambda m: pde.residual(m, x_batch))(model)
        updates, opt_s_new= optimizer.update(grads, opt_s, model)
        model_new= eqx.apply_updates(model, updates)
        return model_new, opt_s_new, loss_val

    def eval_test_l1(model, x_test):
        """只算 L1 误差，也可返回 PDE 残差"""
        pred= model(x_test).reshape(-1)
        true= pde.exact(x_test)
        return jnp.mean(jnp.abs(pred - true))

    step_count=0
    train_loss_list=[]
    test_l1_list=[]
    step_list=[]

    total_steps= steps_per_stage*rad_stages
    log_interval=100

    # (C) 循环 rad_stages
    for stage_i in range(rad_stages):
        # (i) 训练 steps_per_stage
        for _ in range(steps_per_stage):
            fb_model, opt_state, loss_val= train_step(fb_model, opt_state, colloc_pts)
            step_count+=1

            # 周期性记录
            if step_count%log_interval==0 or step_count==total_steps:
                train_loss= loss_val.item()
                l1_err= eval_test_l1(fb_model, test_pts).item()
                print(f"[Stage={stage_i}] step={step_count}, TrainLoss={train_loss:.3e}, TestL1={l1_err:.3e}")
                train_loss_list.append(train_loss)
                test_l1_list.append(l1_err)
                step_list.append(step_count)

        # (ii) 如果不是最后阶段 => RAD 重采样 + 可视化
        if stage_i < rad_stages-1:
            rad_key, key= jax.random.split(key)
            new_pts= sample_by_RAD(
                rad_key, pde, fb_model,
                num_points=rad_num_pts,
                k=k, c=c,
                domain=pde.domain,
                pool_size=rad_pool_size
            )
            colloc_pts= new_pts

            # 可视化
            plt.figure()
            plt.scatter(new_pts[:,0], new_pts[:,1], s=5, c="red", alpha=0.7)
            plt.title(f"Resampled Points after Stage={stage_i+1}")
            plt.xlim([xlo,xhi])
            plt.ylim([ylo,yhi])
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True,ls=":")
            plt.show()

    return fb_model, (step_list, train_loss_list, test_l1_list)


def main():
    # 1) PDE
    pde= Poisson2D_freq()

    # 2) 生成子域(2×2)
    n_sub_per_dim= 2
    overlap= 0.2
    subdomains= generate_subdomains(pde.domain, n_sub_per_dim, overlap)

    # 3) 构造 FBPINN
    mlp_config= dict(in_size=2, out_size=1, width_size=64, depth=2, activation=jax.nn.tanh)
    key= jax.random.PRNGKey(0)
    fb_model= FBPINN(key, subdomains, pde.ansatz, mlp_config, fixed_transition=0.2)

    # 4) 测试点 => 用于测试 L1
    test_n= 40
    test_pts= generate_whole_domain_collocation(pde.domain, test_n)

    # 5) 启动 RAD 训练
    optimizer= optax.adam(1e-3)
    fb_model, (step_list, train_loss_list, test_l1_list)= rad_training_loop(
        key, pde, fb_model, optimizer,
        test_pts= test_pts,
        steps_per_stage=2000,
        rad_stages=5,
        init_num_pts=2500,
        rad_num_pts=2500,
        rad_pool_size=5000,
        k=1.0,
        c=1.0
    )


    # 6) 作图: TrainLoss vs TestL1
    plt.figure()
    plt.plot(step_list, train_loss_list,"-", label="Train PDE Loss")
    plt.plot(step_list, test_l1_list,   "-", label="Test L1 Error")
    plt.yscale("log")
    plt.xlabel("Training Steps")
    plt.ylabel("Value (log scale)")
    plt.title("RAD Training: Loss & L1 Error")
    plt.legend()
    plt.grid(True, ls=":")
    plt.show()

    # （可选）若还想可视化最终解 vs. 真解
    final_pred= fb_model(test_pts).reshape(-1)
    final_true= pde.exact(test_pts).reshape(-1)
    final_err = jnp.abs(final_pred - final_true)
    final_l1  = jnp.mean(final_err)
    print(f"Final L1 error= {final_l1:.3e}")

    final_pred_2d= final_pred.reshape(test_n, test_n)
    final_true_2d= final_true.reshape(test_n, test_n)
    final_err_2d = final_err.reshape(test_n, test_n)

    fig, axs= plt.subplots(1,3, figsize=(15,4), subplot_kw={"aspect":"equal"})
    im0= axs[0].imshow(final_pred_2d, origin="lower", extent=[0,1,0,1], cmap="viridis")
    plt.colorbar(im0, ax=axs[0])
    axs[0].set_title("Pred")

    im1= axs[1].imshow(final_true_2d, origin="lower", extent=[0,1,0,1], cmap="viridis")
    plt.colorbar(im1, ax=axs[1])
    axs[1].set_title("Exact")

    im2= axs[2].imshow(final_err_2d, origin="lower", extent=[0,1,0,1], cmap="viridis")
    plt.colorbar(im2, ax=axs[2])
    axs[2].set_title("Absolute Error")
    plt.suptitle("Final Solution & Error")
    plt.show()


if __name__=="__main__":
    main()
