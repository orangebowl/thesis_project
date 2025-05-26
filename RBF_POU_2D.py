from __future__ import annotations 
import math, jax, jax.numpy as jnp, optax, matplotlib.pyplot as plt
import pathlib
jax.config.update("jax_enable_x64", True)


# ------------------------------------------------------------
# utils
# ------------------------------------------------------------
class Normalizer:
    def __init__(self, min_val, max_val):
        min_val, max_val = jnp.asarray(min_val), jnp.asarray(max_val)
        self.min = min_val
        self.range = jnp.where(max_val == min_val, 1.0, max_val - min_val)

    def transform(self, x):        return (x - self.min) / self.range
    def inverse(self, x_norm):     return x_norm * self.range + self.min


def toy_func(xy):
    x = xy[..., 0]; y = xy[..., 1]
    return jnp.sin(2 * jnp.pi * x**2) * jnp.sin(2 * jnp.pi * y**2)


# ------------------------------------------------------------
# network
# ------------------------------------------------------------
class RBFPOUNet:
    def __init__(self, input_dim, num_centers, key):
        self.input_dim, self.num_centers = input_dim, num_centers
        k1, k2 = jax.random.split(key)
        base = jax.random.uniform(k1, (num_centers, input_dim))
        jitter = 0.02 * jax.random.normal(k2, base.shape)
        self._init_centers = jnp.clip(base + jitter, 0.0, 1.0)
        self._init_widths = 0.15 * jnp.ones((num_centers,))

    def init_params(self):
        return {"centers": self._init_centers, "widths": self._init_widths}

    @staticmethod
    def _gaussian_rbf(d2, w): return jnp.exp(-d2 / (w ** 2 + 1e-12))

    def forward(self, centers, widths, x):
        d2 = jnp.sum((x[:, None, :] - centers[None, :, :]) ** 2, -1)
        log_phi = -d2 / (widths**2 + 1e-12)        # (N,C)
        shift   = jnp.max(log_phi, axis=1, keepdims=True)
        phi     = jnp.exp(log_phi - shift)          # 已经数值稳定
        return phi / jnp.sum(phi, axis=1, keepdims=True)


# ------------------------------------------------------------
# local-poly helpers
# ------------------------------------------------------------
def _design_matrix(x):
    x1, x2 = x[:, 0], x[:, 1]
    return jnp.stack([jnp.ones_like(x1), x1, x2, x1**2, x1 * x2, x2**2], -1)


def fit_local_polynomials_2nd(x, y, w):
    A, y = _design_matrix(x), y[:, None]

    def solve(weights):
        Aw = A * weights[:, None]
        M = A.T @ Aw
        b = (Aw.T @ y).squeeze(-1)
        return jnp.linalg.solve(M + 1e-8 * jnp.eye(6), b)

    return jax.vmap(solve, 1, 0)(w)


def _predict_from_coeffs(x, coeffs, partitions):
    A = _design_matrix(x)
    y_cent = A @ coeffs.T
    return jnp.sum(partitions * y_cent, 1)


# ------------------------------------------------------------
# visualisation helpers (forced title = "epoch = 0")
# ------------------------------------------------------------
def visualize_all_partitions(model, params, grid=200, title=None, save_dir=None):
    xs = jnp.linspace(0, 1, grid)
    xx, yy = jnp.meshgrid(xs, xs)
    pts = jnp.stack([xx.ravel(), yy.ravel()], -1)
    part = model.forward(params["centers"], params["widths"], pts)
    C = part.shape[1]
    cols = int(math.ceil(math.sqrt(C)))
    rows = int(math.ceil(C / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.ravel()
    for i in range(C):
        axes[i].imshow(part[:, i].reshape(grid, grid),
                       origin="lower",
                       extent=[0, 1, 0, 1],
                       cmap="viridis",
                       vmin=0.0,
                       vmax=1.0)
        axes[i].set_title(f"partition {i}")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    for j in range(C, len(axes)):
        axes[j].axis("off")

    # Force the figure title to be "epoch = 0"
    fig.suptitle("epoch = 0", fontsize=14)

    plt.tight_layout()

    if save_dir:
        # Save the image to the specified folder
        save_path = save_dir / "epoch_0_partitions.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved partition visualization to {save_path}")
    else:
        plt.show()


def visualize_final_approximation(model, params, x_train, y_train, num_points=200, save_dir=None):
    part_tr = model.forward(params["centers"], params["widths"], x_train)
    coeffs = fit_local_polynomials_2nd(x_train, y_train, part_tr)

    xs = jnp.linspace(0, 1, num_points)
    xx, yy = jnp.meshgrid(xs, xs)
    pts = jnp.stack([xx.ravel(), yy.ravel()], -1)

    part = model.forward(params["centers"], params["widths"], pts)
    y_pred = _predict_from_coeffs(pts, coeffs, part).reshape(num_points, num_points)
    y_true = toy_func(pts).reshape(num_points, num_points)
    err = jnp.abs(y_true - y_pred)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, z, t in zip(axs, [y_pred, y_true, err],
                        ["prediction", "truth", "abs error"]):
        im = ax.imshow(z, cmap="viridis",
                       origin="lower",
                       extent=[0, 1, 0, 1])
        ax.set_title(t)
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Force the figure title to be "epoch = 0"
    fig.suptitle("epoch = 0", fontsize=14)

    plt.tight_layout()

    if save_dir:
        # Save the final approximation results
        save_path = save_dir / "epoch_0_final_approximation.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved final approximation visualization to {save_path}")
    else:
        plt.show()

    # slice y = 0.5
    xs_slice = jnp.linspace(0, 1, 400)
    pts_slice = jnp.stack([xs_slice, jnp.full_like(xs_slice, 0.5)], -1)
    part_slice = model.forward(params["centers"], params["widths"], pts_slice)
    y_pred_slice = _predict_from_coeffs(pts_slice, coeffs, part_slice)
    y_true_slice = toy_func(pts_slice)

    plt.figure(figsize=(6, 3))
    plt.plot(xs_slice, y_true_slice, "--", label="truth")
    plt.plot(xs_slice, y_pred_slice, label="prediction")
    plt.legend()
    plt.tight_layout()

    if save_dir:
        # Save the slice visualization
        slice_save_path = save_dir / "epoch_0_slice_prediction.png"
        plt.savefig(slice_save_path, dpi=300)
        plt.close()
        print(f"Saved slice visualization to {slice_save_path}")
    else:
        plt.show()


# ------------------------------------------------------------
# training (without width annealing)
# ------------------------------------------------------------
def train_two_phase_lsgd_rbf(
        model, x_train, y_train,
        num_epochs_phase1=6000, num_epochs_phase2=2000,
        lambda_reg=0.1, rho=0.99, n_stag=100,
        lr_phase1=1e-3, lr_phase2=5e-4,
        record_interval=200, viz_interval=None, prints=10):
    """
    这个版本去掉了宽度退火机制
    """
    # ---------- 初始化 ----------
    params = model.init_params()
    snapshots = []

    def maybe_snap(ep):
        if record_interval and ep % record_interval == 0:
            snapshots.append((ep, {"centers": params["centers"].copy(),
                                   "widths": params["widths"].copy()}))

    @jax.jit
    def loss_fn(p, reg_lambda):
        part = model.forward(p["centers"], p["widths"], x_train)
        coeffs = fit_local_polynomials_2nd(x_train, y_train, part)
        pred = _predict_from_coeffs(x_train, coeffs, part)
        return jnp.mean((pred - y_train) ** 2) + reg_lambda * jnp.mean(p["widths"] ** 2)

    valgrad = jax.jit(jax.value_and_grad(loss_fn))

    best_loss, stagn = jnp.inf, 0
    global_ep, total_ep = 0, num_epochs_phase1 + num_epochs_phase2
    log_interval = max(1, total_ep // prints)

    # Create a new directory for saving images
    save_dir = pathlib.Path("visualizations")
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 两阶段循环 ----------
    for epochs, lr, reg0 in [(num_epochs_phase1, lr_phase1, lambda_reg),
                             (num_epochs_phase2, lr_phase2, 0.0)]:
        opt = optax.adam(lr)
        opt_state = opt.init(params)
        reg_lambda = reg0

        for _ in range(epochs):
            # ----- 一步优化 -----
            loss_val, grads = valgrad(params, reg_lambda)
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            loss_val.block_until_ready()

            # ----- 可视化 / 记录 -----
            if viz_interval and global_ep % viz_interval == 0:
                visualize_all_partitions(model, params,
                                         title=f"epoch {global_ep}", save_dir=save_dir)
            maybe_snap(global_ep)

            if global_ep % log_interval == 0:
                print(f"epoch {global_ep:5d}  loss {loss_val:.6e}")

            # ----- 动态调整正则 -----
            if loss_val < best_loss - 1e-12:
                best_loss, stagn = loss_val, 0
            else:
                stagn += 1
            if stagn > n_stag:
                reg_lambda *= rho
                stagn = 0

            global_ep += 1

    return params, snapshots


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
if __name__ == "__main__":
    key = jax.random.PRNGKey(1)
    model = RBFPOUNet(2, 4, key)

    xs = jnp.linspace(0, 1, 40)
    xx, yy = jnp.meshgrid(xs, xs)
    x_raw = jnp.stack([xx.ravel(), yy.ravel()], -1)
    norm = Normalizer(x_raw.min(0), x_raw.max(0))
    x_train = norm.transform(x_raw); y_train = toy_func(x_train)

    params, snaps = train_two_phase_lsgd_rbf(
        model, x_train, y_train,
        record_interval=100,
        viz_interval=500,    # 每 500 epoch 画一次 4-partition 权重热图
        prints=10
    )

    # 最终 PoU & 逼近效果
    visualize_all_partitions(model, params, title="final partitions", save_dir=pathlib.Path("visualizations"))
    visualize_final_approximation(model, params, x_train, y_train, save_dir=pathlib.Path("visualizations"))
