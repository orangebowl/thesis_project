import jax.numpy as jnp


# -------------------------------------------------
# 1. 构建模型
# -------------------------------------------------
def build_model(cfg, key, ansatz, problem):
    """
    cfg   : YAML 转成的对象
    key   : jax.random.PRNGKey
    ansatz: 来自 problem.ansatz
    problem.domain 一定存在，形如 (left, right)
    """
    dom = problem.domain          # 直接拿

    if cfg.model.lower() == "fbpinn":
        from model.fbpinn_model import FBPINN
        return FBPINN(
            dom,
            n_subdomains=cfg.n_subdomains,
            width=cfg.width,
            depth=cfg.depth,
            ansatz_fn=ansatz,
            key=key,
        )
    else:                         # PINN
        from model.pinn_model import PINN
        return PINN(key, ansatz, width=cfg.width, depth=cfg.depth)


# -------------------------------------------------
# 2. 生成采样点
# -------------------------------------------------
def build_collocation(cfg, model, problem):
    left, right = problem.domain

    if cfg.model.lower() == "fbpinn":
        return model.generate_collocation_points(Np=cfg.Np_sub)    # list
    else:
        return jnp.linspace(left, right, cfg.Np_total)             # ndarray
