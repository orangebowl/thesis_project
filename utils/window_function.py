import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle


##############################################################################
# 1) 窗口函数
##############################################################################

import jax
import jax.numpy as jnp


def bump_window_func(xmins_all, xmaxs_all, wmins_all, wmaxs_all, x, eps=1e-9):
    """
    Smooth bump window  — 返回 (N, n_sub) 权重矩阵
    公式：   w(t) = exp( 3 / (t² - 1.001) )      (-1 < t < 1)
            w     = 0                            (|t| ≥ 1)
    其中   t = (x - μ) / σ,   μ = (xmin+xmax)/2,  σ = (xmax-xmin)/2
    """
    # 强制成二维 (N,d)
    x = jnp.atleast_2d(x)
    N, d = x.shape
    n_sub = xmins_all.shape[0]

    # 广播到 (N,n_sub,d)
    x_   = x[:, None, :]                       # (N,1,d)
    xmin = xmins_all[None, :, :]
    xmax = xmaxs_all[None, :, :]

    # 中心与半宽
    mu = (xmin + xmax) / 2.0
    sd = (xmax - xmin) / 2.0 + eps            # 防 0

    # 标准化坐标
    t = (x_ - mu) / sd                        # (N,n_sub,d)

    # bump 核：仅在 (-1,1) 内非零
    inside = jnp.abs(t) < 1.0
    core = jnp.exp(3.0 / (t**2 - 1.001)) / jnp.exp(-3.0)

    w_dim = jnp.where(inside, core, 0.0)

    # 多维相乘 → (N,n_sub)
    w_raw = jnp.prod(w_dim, axis=-1)

    # 如果你想让左右过渡带宽可调，把下两行注释去掉，
    # 并将 wmins_all / wmaxs_all 作为 (left,right) “裁剪”用：
    # mask_left  = (x_ >= xmin + wmins_all[None,:,:])
    # mask_right = (x_ <= xmax - wmaxs_all[None,:,:])
    # w_raw = w_raw * jnp.all(mask_left & mask_right, axis=-1)

    return w_raw


Pi = jnp.pi
def cosine(xmins_all, xmaxs_all,
                           wmins_all, wmaxs_all,     # 占位
                           x, tol: float = 1e-12):
    """
    Cos² window (C² 连续, FBPINN-style)
    输入:
        xmins_all, xmaxs_all : (n_sub, d)
        x                    : (N,d) 或 (d,)
    返回:
        w : (N, n_sub)   已归一化,  ∑_sub w = 1
    """

    # ---------- 保证形状齐全 ----------
    xmins_all = jnp.asarray(xmins_all)
    xmaxs_all = jnp.asarray(xmaxs_all)
    x         = jnp.asarray(x)

    if xmins_all.ndim == 1:                 # (n_sub,) ➔ (n_sub,1)
        xmins_all = xmins_all[:, None]
        xmaxs_all = xmaxs_all[:, None]
    if x.ndim == 1:                         # (d,)     ➔ (1,d)  ← 修正
        x = x[None, :]

    # ---------- 广播到 (N, n_sub, d) ----------
    x_   = x[:, None, :]                    # (N,1,d)
    xmin = xmins_all[None, :, :]            # (1,n_sub,d)
    xmax = xmaxs_all[None, :, :]

    mu = (xmin + xmax) / 2.0
    sd = (xmax - xmin) / 2.0 + tol

    r     = (x_ - mu) / sd                 # 归一化到 [-1,1]
    core  = 0.25 * (1.0 + jnp.cos(Pi * r))**2
    w_dim = jnp.where(jnp.abs(r) <= 1.0, core, 0.0)

    # ---------- 子域原始权重 ----------
    w_raw = jnp.prod(w_dim, axis=-1)        # (N, n_sub)

    # ---------- POU 归一化 ----------
    denom = jnp.maximum(w_raw.sum(axis=1, keepdims=True), 1e-12)
    w = w_raw / denom                       # (N, n_sub)

    return w

def sigmoid(xmins_all, xmaxs_all, wmins_all, wmaxs_all, x, tol=1e-8):
    # Ensure x is 2D
    x = jnp.atleast_2d(x)
    N, d_spatial = x.shape
    n_sub = xmins_all.shape[0]

    # Check shapes
    if not (
        xmins_all.shape == (n_sub, d_spatial)
        and xmaxs_all.shape == (n_sub, d_spatial)
        and wmins_all.shape == (n_sub, d_spatial)
        and wmaxs_all.shape == (n_sub, d_spatial)
    ):
        raise ValueError("Shape mismatch in xmins_all/xmaxs_all/wmins_all/wmaxs_all.")

    # Sigmoid sharpness parameter
    t = jnp.log((1 - tol) / tol)

    # Left-side transition
    mu_min = xmins_all + wmins_all / 2.0
    sd_min = wmins_all / (2.0 * t)

    # Right-side transition
    mu_max = xmaxs_all - wmaxs_all / 2.0
    sd_max = wmaxs_all / (2.0 * t)

    # Broadcast x
    x_exp = x[:, None, :]   # shape: (N, 1, d_spatial)

    # Sigmoid arguments
    left_sig_arg  = (x_exp - mu_min)  / sd_min
    right_sig_arg = (mu_max - x_exp) / sd_max

    # Compute sigmoids
    left_sig  = jax.nn.sigmoid(left_sig_arg)   # (N, n_sub, d_spatial)
    right_sig = jax.nn.sigmoid(right_sig_arg)  # (N, n_sub, d_spatial)

    # Multiply across dimensions
    prod_sig = left_sig * right_sig            # (N, n_sub, d_spatial)
    w_raw = jnp.prod(prod_sig, axis=-1)        # (N, n_sub)

    return w_raw


def plot_window_weights_2d(domain_bounds, subdomains,
                           transition, n_points=50, tol_plot=1e-3):

    (x_lo, y_lo), (x_hi, y_hi) = domain_bounds

    x_vals = jnp.linspace(x_lo, x_hi, n_points)
    y_vals = jnp.linspace(y_lo, y_hi, n_points)
    X, Y = jnp.meshgrid(x_vals, y_vals, indexing="ij")
    XY_flat = jnp.column_stack([X.ravel(), Y.ravel()])

    xmins = jnp.stack([s[0] for s in subdomains])  # (n_sub, 2)
    xmaxs = jnp.stack([s[1] for s in subdomains])
    wmins = jnp.full_like(xmins, transition)
    wmaxs = jnp.full_like(xmaxs, transition)

    # compute weights
    w_raw = sigmoid(xmins, xmaxs, wmins, wmaxs, XY_flat, tol=tol_plot)
    n_sub = len(subdomains)

    # weight map
    w_maps = [w_raw[:, i].reshape(n_points, n_points) for i in range(n_sub)]
    w_sum = jnp.sum(w_raw, axis=1).reshape(n_points, n_points)


    fig, axs = plt.subplots(1, n_sub + 1,
                            figsize=(4.5*(n_sub + 1), 5),
                            subplot_kw=dict(aspect="equal"))
    if n_sub == 1:
        axs = [axs]

    fig.suptitle("2D Window Weights Visualization", fontsize=15, y=1.02)

    norm_sub = mcolors.Normalize(vmin=0.0, vmax=1.0)
    for i, ax in enumerate(axs[:-1]):
        im = ax.imshow(
            w_maps[i],
            extent=[x_lo, x_hi, y_lo, y_hi],
            origin="lower",
            cmap="coolwarm",
            norm=norm_sub
        )
        ax.set_title(f"Subdomain {i}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        #  colorbar
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("weight", rotation=90)

    # Sum of Weights
    ax_sum = axs[-1]
    vmax_sum = float(w_sum.max())  
    norm_sum = mcolors.Normalize(vmin=0.0, vmax=vmax_sum)
    im_sum = ax_sum.imshow(
        w_sum,
        extent=[x_lo, x_hi, y_lo, y_hi],
        origin="lower",
        cmap="viridis",
        norm=norm_sum
    )
    ax_sum.set_title("Sum of Weights")
    ax_sum.set_xlabel("x")
    ax_sum.set_ylabel("y")

    cb_sum = fig.colorbar(im_sum, ax=ax_sum, fraction=0.046, pad=0.04)
    cb_sum.set_label("sum of weights", rotation=90)

    plt.tight_layout()
    plt.show()


def plot_window_weights_1d(domain_bounds, subdomains,
                           transition, n_points=500, tol_plot=1e-3):

    x_plot = jnp.linspace(domain_bounds[0][0], domain_bounds[1][0], n_points).reshape(-1, 1)
    xmins = jnp.stack([s[0] for s in subdomains])
    xmaxs = jnp.stack([s[1] for s in subdomains])
    wmins = jnp.full_like(xmins, transition)
    wmaxs = jnp.full_like(xmaxs, transition)
    w = sigmoid(xmins, xmaxs, wmins, wmaxs, x_plot, tol=tol_plot)

    fig, ax = plt.subplots(figsize=(10, 4))

    for j in range(w.shape[1]):
        ax.plot(x_plot.squeeze(), w[:, j], label=f"Subdomain {j}")
    # sum of weights
    ax.plot(x_plot.squeeze(), jnp.sum(w, axis=1), "k--", label="Sum of weights")

    y_off, bar_h = -0.06, 0.03
    for j, (xmin, xmax) in enumerate(subdomains):
        ax.add_patch(Rectangle((xmin[0], y_off),
                               xmax[0] - xmin[0], bar_h,
                               color=f"C{j}", alpha=0.4))
        ax.text((xmin[0] + xmax[0]) / 2,
                y_off + bar_h / 2,
                f"{j}",
                ha="center", va="center",
                fontsize=8, color="white")

    dom_lo, dom_hi = domain_bounds[0][0], domain_bounds[1][0]
    ax.axvline(dom_lo, y_off - 0.02, 1.02, ls=":", color="grey")
    ax.axvline(dom_hi, y_off - 0.02, 1.02, ls=":", color="grey")
    ax.text(dom_lo, y_off - 0.03, f"{dom_lo:.2f}",
            ha="center", va="top", fontsize=8, color="grey")
    ax.text(dom_hi, y_off - 0.03, f"{dom_hi:.2f}",
            ha="center", va="top", fontsize=8, color="grey")

    ax.set_ylim(y_off - 0.02, 1.05)
    ax.set_title(f"1D window weights (transition={transition})")
    ax.set_xlabel("x")
    ax.set_ylabel("weight")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.show()



if __name__ == "__main__":
    from data_utils import generate_subdomains

    overlap = 0.2
    tol = 1e-8

    # 2D test
    print("--- 2D TEST ---")
    domain_2d = (jnp.array([0., 0.]), jnp.array([1., 1.]))
    subs_2d = generate_subdomains(domain_2d, n_sub_per_dim=2, overlap=overlap)
    print(f"2D number of subdomains: {len(subs_2d)}")

    # test weights on points
    x_test_2d = jnp.array([
        [0.25, 0.25],
        [0.75, 0.75],
        [0.50, 0.50]
    ])
    xmins_2d = jnp.stack([s[0] for s in subs_2d])
    xmaxs_2d = jnp.stack([s[1] for s in subs_2d])
    wmins_2d = jnp.full_like(xmins_2d, overlap)
    wmaxs_2d = jnp.full_like(xmaxs_2d, overlap)
    w2d = cosine(xmins_2d, xmaxs_2d, wmins_2d, wmaxs_2d, x_test_2d)
    print("Weights at sample pts (2D):\n", w2d)

    plot_window_weights_2d(
        domain_bounds=domain_2d,
        subdomains=subs_2d,
        transition=overlap,
        n_points=100,
        tol_plot=tol
    )

    # 1D test
    print("\n--- 1D TEST ---")
    domain_1d = (jnp.array([0.]), jnp.array([3.]))
    subs_1d = generate_subdomains(domain_1d, n_sub_per_dim=4, overlap=overlap)
    print(f"1D 子域数量: {len(subs_1d)}")

    plot_window_weights_1d(domain_1d, subs_1d, overlap, tol_plot=tol)

    # 1D self defined domain
    print("\n--- 1D CUSTOM SUBDOMAINS TEST ---")
    domain_1d_custom = (jnp.array([0.]), jnp.array([3.]))
    subdomains_custom = [
        (jnp.array([-1.6]), jnp.array([1.6])),
        (jnp.array([1.5]),  jnp.array([2.4])),
        (jnp.array([2.3]),  jnp.array([2.9])),
        (jnp.array([2.8]),  jnp.array([3.2]))
    ]
    transition_width_custom = 0.1
    plot_window_weights_1d(domain_1d_custom,
                           subdomains_custom,
                           transition_width_custom,
                           n_points=400,
                           tol_plot=tol)

    print("All tests completed.")
