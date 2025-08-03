"""
Residual-based Adaptive sampling (RAD) demo with tqdm progress bar.
"""
import os, sys, copy
import jax, jax.numpy as jnp, optax, equinox as eqx
import matplotlib.pyplot as plt
from tqdm import trange                    #  ← 进度条

# ---------------------------------------------------------------------
# 路径 & 模块
# ---------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from physics.problems   import Poisson2D_freq
from model.fbpinn_model import FBPINN
from utils.data_utils   import generate_subdomains, generate_collocation
from vis.vis_2d         import visualize_2d

# ---------------------------------------------------------------------
# 点-级残差
# ---------------------------------------------------------------------
def pointwise_residual(pde, model, xy):
    def u_fn(pt): return model(pt).squeeze()
    hess = jax.jacfwd(jax.jacrev(u_fn))
    lap  = jax.vmap(lambda pt: jnp.trace(hess(pt)))(xy)
    return jnp.abs(-lap - pde.rhs(xy))      # (N,)

# ---------------------------------------------------------------------
# RAD 采样
# ---------------------------------------------------------------------
def rad_sample(key, pde, model, *, n_draw, pool_size, k=3.0, c=1.0):
    #pool_size = max(pool_size, n_draw * 2)
    (lo, hi)  = pde.domain
    kx, ky, key = jax.random.split(key, 3)
    xs = jax.random.uniform(kx, (pool_size,),
                            minval=lo[0], maxval=hi[0])
    ys = jax.random.uniform(ky, (pool_size,),
                            minval=lo[1], maxval=hi[1])
    pool = jnp.column_stack([xs, ys])

    eps  = pointwise_residual(pde, model, pool)
    prob = eps**k / jnp.maximum(jnp.mean(eps**k), 1e-12) + c
    prob = prob / prob.sum()
    idx  = jax.random.choice(key, pool_size, (n_draw,), p=prob, replace=False)
    return pool[idx]

# Plot help 
# plot the collocation points at each stage
def plot_colloc_points(colloc, domain, stage_id, save_dir="results"):
    """
    在给定的二维区域 domain 上，绘制 colloc 点分布，并以 stage_id 结尾保存图片。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure()
    plt.scatter(colloc[:, 0], colloc[:, 1], s=10, alpha=0.5, edgecolors='k')
    plt.xlim(domain[0][0], domain[1][0])
    plt.ylim(domain[0][1], domain[1][1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Stage {stage_id + 1} Collocation Points (RAD)")
    plt.savefig(os.path.join(save_dir, f"rad_colloc_stage_{stage_id + 1}.png"))
    plt.close()

#FBPINN training with RAD
def train_fbpinn_rad(
    *, key, pde, fbmodel,
    stages, steps_per_stage, n_init, n_rad, pool_size, xdim,
    lr=1e-3, eval_every=200,
):
    # 初始均匀采样
    colloc = generate_collocation(pde.domain, n_init, "halton")
    opt    = optax.adam(lr)
    opt_st = opt.init(eqx.filter(fbmodel, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, xb):
        def loss_fn(m):
            res_sq = jax.vmap(m.residual_fn, (None, 0))(m, xb)
            return jnp.mean(res_sq)
        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = opt.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val

    def l1(model):
        pts = generate_collocation(pde.domain, 80, "grid")
        return jnp.mean(jnp.abs(model(pts).squeeze() - pde.exact(pts)))

    total_steps = stages * steps_per_stage
    loss_hist, l1_steps, l1_hist = [], [], []
    stage_boundary = steps_per_stage     # 下一次重采样的 step 编号
    stage_id = 0

    pbar = trange(total_steps, desc="RAD-FBPINN", dynamic_ncols=True)
    for global_step in pbar:
        fbmodel, opt_st, loss_val = step(fbmodel, opt_st, colloc)
        loss_hist.append(loss_val)

        if (global_step + 1) % eval_every == 0 or global_step == total_steps - 1:
            l1_val = float(l1(fbmodel))
            l1_hist.append(l1_val)
            l1_steps.append(global_step + 1)
            pbar.set_postfix(loss=f"{loss_val:.2e}", l1=f"{l1_val:.2e}")
        else:
            pbar.set_postfix(loss=f"{loss_val:.2e}")

        # ----- 到达 stage 末尾 → RAD 重采样 ----- #
        if global_step + 1 == stage_boundary and stage_id < stages - 1:
            key, sub = jax.random.split(key)
            colloc = rad_sample(
                sub, pde, fbmodel,
                n_draw=n_rad ** xdim, pool_size=pool_size
            )
            # 画出此阶段采样到的新 colloc 点分布
            plot_colloc_points(colloc, pde.domain, stage_id, save_dir="results")

            stage_id += 1
            stage_boundary += steps_per_stage

    return fbmodel, jnp.array(loss_hist), jnp.array(l1_steps), jnp.array(l1_hist)

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    problem = Poisson2D_freq()
    domain  = problem.domain
    xdim    = 1 if domain[0].size == 1 else 2
    overlap = 0.2

    subdomains = generate_subdomains(domain, n_sub_per_dim=2, overlap=overlap)
    mlp_c = dict(in_size=xdim, out_size=1, width_size=64, depth=2,
                 activation=jax.nn.tanh)

    key = jax.random.PRNGKey(1)
    fb  = FBPINN(
        key              = key,
        subdomains       = subdomains,
        mlp_config       = mlp_c,
        ansatz           = problem.ansatz,
        residual_fn      = problem.residual,
        fixed_transition = overlap,
        window_fn=None,
    )

    model, loss_hist, l1_steps, l1_hist = train_fbpinn_rad(
        key             = key,
        pde             = problem,
        fbmodel         = fb,
        stages          = 10,
        steps_per_stage = 3000,
        n_init          = 50,
        n_rad           = 50,
        pool_size       = 10000,          # 自动扩展
        xdim            = xdim,
        lr              = 1e-3,
        eval_every      = 200,
    )

    # ---------- 可视化最终预测结果 ---------- #
    Nx = Ny = 80
    gx = jnp.linspace(domain[0][0], domain[1][0], Nx)
    gy = jnp.linspace(domain[0][1], domain[1][1], Ny)
    mesh = jnp.stack(jnp.meshgrid(gx, gy, indexing="ij"), -1)
    grid = mesh.reshape(-1, xdim)

    u_pred  = model(grid).reshape(Nx, Ny)
    u_exact = problem.exact(grid).reshape(Nx, Ny)

    visualize_2d(model,
        gx, gy, u_pred, u_exact,
        loss_hist, l1_steps, l1_hist,
        save_dir="results", title_prefix="RAD-FBPINN"
    )

    print("Figures saved to ./results")
    
if __name__ == "__main__":
    main()
