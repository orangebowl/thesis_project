import jax,optax
import equinox as eqx
import jax.numpy as jnp
from model.fbpinn_model import FBPINN_PoU
from vis.vis_pou import plot_colloc_points
from tqdm import trange

def pointwise_residual_pou(problem, model, xy):
    """
    计算 FBPINN_PoU 模型在每个点 xy 处的物理残差的绝对值。
    (修正版：不再依赖 get_raw_residuals，而是自行计算)
    """
    # 1. 定义一个辅助函数，用于获取模型在单点的标量输出
    def u_fn(point):
        return model(point).squeeze()

    # 2. JIT-编译Hessian计算，并用vmap并行处理整个批次
    #    Hessian(u) = ∇(∇u)
    hessian_fn = jax.jacfwd(jax.jacrev(u_fn))
    #    vmap并行计算，得到每个点的Hessian矩阵
    hessians = jax.vmap(hessian_fn)(xy)
    
    # 3. 拉普拉斯算子是Hessian矩阵的迹
    #    Δu = trace(Hessian(u))
    laplacians = jnp.trace(hessians, axis1=-2, axis2=-1)

    # 4. 计算残差 |Δu + f| (对于PDE -Δu = f)
    #    problem.rhs(xy) 对应 f(x,y)
    raw_residuals = laplacians + problem.rhs(xy)
    
    return jnp.abs(raw_residuals)

def rad_sample(key, problem, model, *, n_draw, pool_size, k=3.0, c=1.0):
    """
    根据模型残差进行自适应采样。(此函数无需修改)
    """
    (lo, hi) = problem.domain
    pool = jax.random.uniform(key, (pool_size, 2), minval=lo, maxval=hi)
    
    key, sub_key = jax.random.split(key)
    
    res_vals = pointwise_residual_pou(problem, model, pool)
    prob = res_vals**k / jnp.mean(res_vals**k) + c
    prob = prob / prob.sum()
    
    idx = jax.random.choice(sub_key, pool_size, (n_draw,), p=prob, replace=False)
    return pool[idx]

def train_fbpinn_pou_rad(
    *,
    key: jax.Array,
    base_model: FBPINN_PoU,          # ① 已经训练过的固定窗口模型
    pou_model:  FBPINN_PoU,          # ② 要训练的“learned-PoU” 模型
    problem:    object,
    # RAD
    stages: int,
    steps_per_stage: int,
    n_colloc: int,
    pool_size: int,
    # 优化 / 评估
    lr: float,
    x_test: jax.Array,
    u_exact: callable,
    eval_every: int = 500,
):
    # ── 1) 以 pou_model 拆分 trainable / static ─────────────────────────
    params, static = eqx.partition(pou_model, eqx.is_array)
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    # ── 2) JIT helpers ──────────────────────────────────────────────────
    @eqx.filter_jit
    def loss_fn(p, xb):
        return problem.residual(model=eqx.combine(p, static), xy=xb)

    @eqx.filter_jit
    def step(p, o, xb):
        loss, g = jax.value_and_grad(loss_fn)(p, xb)
        u, o = opt.update(g, o)              # optax.adam
        p = eqx.apply_updates(p, u)
        return p, o, loss

    @eqx.filter_jit
    def l1_fn(p):
        m = eqx.combine(p, static)
        pred = jax.vmap(m)(x_test).squeeze()
        return jnp.mean(jnp.abs(pred - u_exact(x_test).squeeze()))

    # ── 3) 初始 collocation —— 用 base_model 的残差抽样 ──────────────────
    key, sub = jax.random.split(key)
    colloc = rad_sample(
        sub, problem, base_model,
        n_draw=n_colloc, pool_size=pool_size
    )
    plot_colloc_points(colloc, problem.domain, stage_id=0)

    # ── 4) 训练循环 ─────────────────────────────────────────────────────
    loss_hist, l1_hist, l1_steps = [], [], []
    total = stages * steps_per_stage
    bar   = trange(total, desc="RAD-PoU", dynamic_ncols=True)

    for g in bar:
        params, opt_state, loss = step(params, opt_state, colloc)
        loss_hist.append(float(loss))

        if (g + 1) % eval_every == 0 or g + 1 == total:
            l1 = float(l1_fn(params))
            l1_hist.append(l1); l1_steps.append(g + 1)
            bar.set_postfix(loss=f"{loss:.2e}", L1=f"{l1:.2e}")
        else:
            bar.set_postfix(loss=f"{loss:.2e}")

        # 每到一个 stage 末尾，用 *当前 pou 模型* 残差再采样一次
        if (g + 1) % steps_per_stage == 0 and (g + 1) < total:
            stage = (g + 1) // steps_per_stage
            key, sub = jax.random.split(key)
            curr_model = eqx.combine(params, static)
            colloc = rad_sample(
                sub, problem, curr_model,
                n_draw=n_colloc, pool_size=pool_size
            )
            plot_colloc_points(colloc, problem.domain, stage_id=stage)

    final_model = eqx.combine(params, static)
    return final_model, jnp.array(loss_hist), (jnp.array(l1_steps), jnp.array(l1_hist))