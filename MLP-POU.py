from __future__ import annotations
import math, jax, jax.numpy as jnp, optax, matplotlib.pyplot as plt
import pathlib

jax.config.update("jax_enable_x64", True)

# 创建保存图片的文件夹
SAVE_DIR = pathlib.Path("visualizations")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
#  toy function
# ------------------------------------------------------------------
def toy_func(xy: jnp.ndarray) -> jnp.ndarray:
    x, y = xy[..., 0], xy[..., 1]
    return jnp.sin(2 * jnp.pi * x**2) * jnp.sin(2 * jnp.pi * y**2)


# ------------------------------------------------------------------
#  MLP-PoU
# ------------------------------------------------------------------
def glorot(k, shape):
    fan_in, fan_out = shape
    limit = jnp.sqrt(6 / (fan_in + fan_out))
    return jax.random.uniform(k, shape, minval=-limit, maxval=limit)

class MLPPOUNet:
    """A small MLP that outputs softmax weights (PoU)."""
    def __init__(self,
                 input_dim: int,
                 num_experts: int,
                 hidden: tuple[int, ...] = (32, 32, 32, 32),
                 key: jax.random.KeyArray | None = None):
        self.num_experts = num_experts
        key = jax.random.PRNGKey(0) if key is None else key
        keys = jax.random.split(key, len(hidden) + 1)

        p = {}
        in_dim = input_dim
        for i, h in enumerate(hidden):
            p[f"W{i}"] = glorot(keys[i], (in_dim, h))
            p[f"b{i}"] = jnp.zeros((h,))
            in_dim = h
        p[f"W{len(hidden)}"] = glorot(keys[-1], (in_dim, num_experts))
        p[f"b{len(hidden)}"] = jnp.zeros((num_experts,))
        self._init_params = p

    def init_params(self) -> dict[str, jnp.ndarray]:
        return {k: v.copy() for k, v in self._init_params.items()}

    @staticmethod
    def forward(params: dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        h = x
        n_layer = (len(params) // 2) - 1
        for i in range(n_layer):
            h = jnp.tanh(h @ params[f"W{i}"] + params[f"b{i}"])
        logits = h @ params[f"W{n_layer}"] + params[f"b{n_layer}"]
        return jax.nn.softmax(logits, axis=-1)             # (N, C)


# ------------------------------------------------------------------
#  local polynomial fitting
# ------------------------------------------------------------------
def _design_matrix(x: jnp.ndarray) -> jnp.ndarray:
    x1, x2 = x[:, 0], x[:, 1]
    return jnp.stack([jnp.ones_like(x1), x1, x2, x1**2, x1 * x2, x2**2], -1)

def fit_local_polynomials_2nd(x, y, w, lam: float = 0.0):
    A, y = _design_matrix(x), y[:, None]

    def solve(weights):
        Aw = A * weights[:, None]
        M  = A.T @ Aw
        b  = (Aw.T @ y).squeeze(-1)
        return jnp.linalg.solve(M + lam * jnp.eye(6), b)

    return jax.vmap(solve, 1, 0)(w)        # (C,6)

def _predict_from_coeffs(x, coeffs, partitions):
    A = _design_matrix(x)
    y_cent = A @ coeffs.T                  # (N, C)
    return jnp.sum(partitions * y_cent, 1)


# ------------------------------------------------------------------
#  可视化函数：保存图片到 SAVE_DIR
# ------------------------------------------------------------------
def visualize_all_partitions(model, params, grid=200, title=None, save_name=None):
    """
    可视化 partition; 如果传入 save_name，则保存为 {save_name}.png
    """
    xs = jnp.linspace(0, 1, grid)
    xx, yy = jnp.meshgrid(xs, xs)
    p = model.forward(params, jnp.stack([xx.ravel(), yy.ravel()], -1))
    C = p.shape[1]
    cols = int(math.ceil(math.sqrt(C)))
    rows = int(math.ceil(C/cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3*cols,3*rows))
    axes = axes.ravel()
    for i in range(C):
        axes[i].imshow(
            p[:, i].reshape(grid, grid),
            origin="lower",
            extent=[0,1,0,1],
            cmap="viridis",
            vmin=0,
            vmax=1
        )
        axes[i].set_title(f"partition {i}")
        axes[i].set_xticks([]); axes[i].set_yticks([])
    for j in range(C, len(axes)):
        axes[j].axis("off")

    if title:
        fig.suptitle(title)
    plt.tight_layout()

    if save_name is not None:
        save_path = SAVE_DIR / f"{save_name}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Partitions saved to: {save_path}")
    else:
        # 如果不需要保存，可以选择关闭或显示
        plt.show()


def visualize_final_approx(model, params, x_train, y_train,
                           num_points=200, save_name_prefix="final_approx"):
    """
    最终逼近效果可视化
    """
    part_tr = model.forward(params, x_train)
    coeffs  = fit_local_polynomials_2nd(x_train, y_train, part_tr, lam=0.0)

    xs = jnp.linspace(0,1,num_points)
    xx, yy = jnp.meshgrid(xs,xs)
    pts = jnp.stack([xx.ravel(), yy.ravel()], -1)
    part = model.forward(params, pts)
    y_pred = _predict_from_coeffs(pts, coeffs, part).reshape(num_points, num_points)
    y_true = toy_func(pts).reshape(num_points, num_points)
    err    = jnp.abs(y_true - y_pred)

    fig, ax = plt.subplots(1,3, figsize=(12,4))
    for a,z,t in zip(ax,[y_pred,y_true,err],["prediction","truth","abs error"]):
        im = a.imshow(z, cmap="viridis", origin="lower", extent=[0,1,0,1])
        a.set_title(t)
        plt.colorbar(im, ax=a, shrink=0.8)
    plt.tight_layout()

    # 保存三合一对比图
    save_path = SAVE_DIR / f"{save_name_prefix}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Final approximation saved to: {save_path}")


# ------------------------------------------------------------------
#  internal: one-phase LSGD (第二版本, JIT编译一次)
# ------------------------------------------------------------------
def _run_lsgd(
    model,
    params,
    x,
    y,
    n_epochs: int,
    lr: float,
    lam_init: float,
    rho: float,
    n_stag: int,
    viz_interval: int | None = None,
    prints: int = 10,
    start_epoch: int = 0
):
    """
    第二版本：先 JIT 编译一次，之后循环中只更新params、重新计算loss。
    lam 用 jnp.array 来方便后续动态衰减
    """
    lam = jnp.array(lam_init)
    best_loss, stag = jnp.inf, 0
    log_int = max(1, n_epochs // prints)

    @jax.jit
    def loss_fn(p, lam_):
        part   = model.forward(p, x)
        coeffs = fit_local_polynomials_2nd(x, y, part, lam_)
        pred   = _predict_from_coeffs(x, coeffs, part)
        return jnp.mean((pred - y)**2)

    # 用于计算梯度
    valgrad = jax.jit(lambda p, lam_: jax.value_and_grad(
        lambda pp: loss_fn(pp, lam_))(p))

    # 优化器
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    ep = start_epoch

    # ---- 先行一次编译 ----
    print("⏳ JIT compiling first step ...")
    loss_val, grads = valgrad(params, lam)
    print("✅ compile done, start training")

    # ---- 训练循环 ----
    for _ in range(n_epochs):
        # 1) 更新参数
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        # 2) 重新计算 loss & grads
        loss_val, grads = valgrad(params, lam)

        # 可视化
        if viz_interval and ep % viz_interval == 0:
            # 将 epoch 号拼接到文件名中，避免覆盖
            save_name = f"epoch_{ep:05d}"
            visualize_all_partitions(model, params,
                                     title=f"epoch_{ep}",
                                     save_name=save_name)

        # 打印
        if ep % log_int == 0:
            print(f"epoch {ep:6d} | loss {loss_val:.6e} | λ {float(lam):.1e}")

        # 动态调整 (early stopping / 正则衰减)
        if loss_val < best_loss - 1e-12:
            best_loss, stag = loss_val, 0
        else:
            stag += 1
        if stag > n_stag:
            lam = lam * rho
            stag = 0

        ep += 1

    return params, ep


# ------------------------------------------------------------------
#  Two-phase-LSGD trainer (第二版本) + 保存图片
# ------------------------------------------------------------------
def train_two_phase(model: MLPPOUNet, x_train, y_train,
                    n_pre: int = 6000, n_post: int = 2000,
                    lr_pre: float = 1e-3, lr_post: float = 5e-4,
                    lam_init: float = 0.1, rho: float = 0.99,
                    n_stag: int = 200,
                    viz_interval: int | None = 500,
                    prints: int = 10):
    """
    第二版本的 2-phase LSGD:
      - Phase 1: lam_init > 0
      - Phase 2: lam = 0
    """
    # 初始化网络参数
    params = model.init_params()

    # --- Phase-1 : λ > 0 ---
    params, ep = _run_lsgd(
        model, params, x_train, y_train,
        n_epochs=n_pre,
        lr=lr_pre,
        lam_init=lam_init,
        rho=rho,
        n_stag=n_stag,
        viz_interval=viz_interval,
        prints=prints,
        start_epoch=0
    )

    # --- Phase-2 : λ = 0 ---
    params, _ = _run_lsgd(
        model, params, x_train, y_train,
        n_epochs=n_post,
        lr=lr_post,
        lam_init=0.0,
        rho=1.0,         # 不再衰减
        n_stag=n_stag,
        viz_interval=viz_interval,
        prints=prints,
        start_epoch=ep
    )

    # 训练结束再用 λ=0 解最终多项式系数
    part   = model.forward(params, x_train)
    coeffs = fit_local_polynomials_2nd(x_train, y_train, part, lam=0.0)
    return params, coeffs


# ------------------------------------------------------------------
#  main
# ------------------------------------------------------------------
if __name__ == "__main__":
    key = jax.random.PRNGKey(1)
    model = MLPPOUNet(input_dim=2, num_experts=4, key=key)

    xs = jnp.linspace(0, 1, 40)
    xx, yy = jnp.meshgrid(xs, xs)
    x_train = jnp.stack([xx.ravel(), yy.ravel()], -1)
    y_train = toy_func(x_train)

    # 调用第二版本的train_two_phase，并把所有图片保存到 visualization 文件夹
    params, coeffs = train_two_phase(
        model, x_train, y_train,
        n_pre=6000,
        n_post=2000,
        lr_pre=1e-3,
        lr_post=5e-4,
        lam_init=1e-3,
        rho=0.99,
        n_stag=100,
        viz_interval=500,  # 每 500 epoch 保存一次 partition 图
        prints=10
    )

    # 最终分区 & 逼近效果
    visualize_all_partitions(model, params,
                             title="final partitions (Two-phase LSGD)",
                             save_name="final_partitions")

    visualize_final_approx(model, params, x_train, y_train,
                           save_name_prefix="final_approximation")
