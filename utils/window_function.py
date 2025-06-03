import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

from utils.data_utils import generate_subdomains


##############################################################################
# 1) 三种窗口函数
##############################################################################

def bump_window_func(xmins_all, xmaxs_all, wmins_all, wmaxs_all, x, eps=1e-9):
    """
    Smooth bump window (非归一化的“帽子”函数)，返回 (N, n_sub) 权重矩阵。
    公式：   w(t) = exp( 3 / (t² - 1.001) ),  对 |t|<1； 否则为0
          其中   t = (x - μ) / σ,   μ = (xmin+xmax)/2,  σ = (xmax-xmin)/2
    注意：这里不做归一化，可根据需要在外部统一做 ∑ 归一化。
    """
    x = jnp.atleast_2d(x)  # 保证 x 形状 (N,d)
    N, d = x.shape
    n_sub = xmins_all.shape[0]

    # 广播到 (N,n_sub,d)
    x_   = x[:, None, :]              
    xmin = xmins_all[None, :, :]
    xmax = xmaxs_all[None, :, :]

    mu = (xmin + xmax) / 2.0
    sd = (xmax - xmin) / 2.0 + eps
    t  = (x_ - mu) / sd

    inside = jnp.abs(t) < 1.0
    core   = jnp.exp(3.0 / (t**2 - 1.001)) / jnp.exp(-3.0)
    w_dim  = jnp.where(inside, core, 0.0)

    # 多维相乘，得到 (N, n_sub)
    w_raw = jnp.prod(w_dim, axis=-1)
    return w_raw


import jax.numpy as jnp
Pi = jnp.pi
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



def sigmoid(xmins_all, xmaxs_all,
                   wmins_all, wmaxs_all,
                   x, tol: float = 1e-8):
    """
    Pair-sigmoid window (C^1).  已完成 POU 归一化.
    """
    xmins_all = jnp.asarray(xmins_all)
    xmaxs_all = jnp.asarray(xmaxs_all)
    wmins_all = jnp.asarray(wmins_all)
    wmaxs_all = jnp.asarray(wmaxs_all)
    x         = jnp.asarray(x)

    if xmins_all.ndim == 1:              # (n_sub,) → (n_sub,1)
        for arr in (xmins_all, xmaxs_all, wmins_all, wmaxs_all):
            arr = arr[:, None]
    if x.ndim == 1:
        x = x[None, :]

    x_   = x[:, None, :]                 # (N,1,d)
    xmin = xmins_all[None, :, :]
    xmax = xmaxs_all[None, :, :]
    wmin = wmins_all[None, :, :]
    wmax = wmaxs_all[None, :, :]

    # t ≈ log((1-ε)/ε) 控制转折陡峭度
    t = jnp.log((1 - tol) / tol)

    # sig(x) = 1 / (1 + exp(-(x-μ)/σ))
    mu_min = xmin + wmin / 2.0
    sd_min = wmin / (2.0 * t) + tol
    mu_max = xmax - wmax / 2.0
    sd_max = wmax / (2.0 * t) + tol

    left  = jax.nn.sigmoid((x_ - mu_min) / sd_min)
    right = jax.nn.sigmoid((mu_max - x_) / sd_max)

    w_dim = left * right                # (N,n_sub,d)
    w_raw = jnp.prod(w_dim, axis=-1)    # (N,n_sub)

    denom = jnp.maximum(w_raw.sum(axis=1, keepdims=True), 1e-12)
    return w_raw / denom

##############################################################################
# 2) 通用的 1D/2D 绘图函数
##############################################################################

def plot_window_weights_1d(window_func, domain_bounds, subdomains,
                           transition, n_points=300, tol_plot=1e-3):
    """
    1D 绘图: 传入具体 window_func(bump / cosine / sigmoid)，并绘制子域权重。
    """
    lo, hi = domain_bounds
    x_plot = jnp.linspace(lo[0], hi[0], n_points).reshape(-1, 1)

    xmins = jnp.stack([s[0] for s in subdomains])
    xmaxs = jnp.stack([s[1] for s in subdomains])
    wmins = jnp.full_like(xmins, transition)
    wmaxs = jnp.full_like(xmaxs, transition)

    # 计算原始权重
    w_raw = window_func(xmins, xmaxs, wmins, wmaxs, x_plot, tol_plot)
    # 对非归一函数做一次 sum-normalize
    denom = jnp.maximum(w_raw.sum(axis=1, keepdims=True), 1e-12)
    w = w_raw / denom

    fig, ax = plt.subplots(figsize=(10, 4))
    for j in range(w.shape[1]):
        ax.plot(x_plot.squeeze(), w[:, j], label=f"Subdomain {j}")

    # sum of weights
    ax.plot(x_plot.squeeze(), jnp.sum(w, axis=1), "k--", label="Sum of weights")

    # 在 x 轴下方画子域区间示意
    y_off, bar_h = -0.06, 0.03
    for j, (xmin, xmax) in enumerate(subdomains):
        ax.add_patch(Rectangle((xmin[0], y_off),
                               xmax[0] - xmin[0], bar_h,
                               color=f"C{j}", alpha=0.4))
        ax.text((xmin[0] + xmax[0]) / 2,
                y_off + bar_h / 2,
                f"{j}", ha="center", va="center",
                fontsize=8, color="white")

    # 画出总 domain
    ax.axvline(lo[0], y_off - 0.02, 1.02, ls=":", color="grey")
    ax.axvline(hi[0], y_off - 0.02, 1.02, ls=":", color="grey")
    ax.text(lo[0], y_off - 0.03,  f"{lo[0]:.2f}",
            ha="center", va="top", fontsize=8, color="grey")
    ax.text(hi[0], y_off - 0.03,  f"{hi[0]:.2f}",
            ha="center", va="top", fontsize=8, color="grey")

    ax.set_ylim(y_off - 0.02, 1.05)
    ax.set_title(f"1D window weights [{window_func.__name__}] (transition={transition})")
    ax.set_xlabel("x")
    ax.set_ylabel("weight")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.show()


def plot_window_weights_2d(window_func, domain_bounds, subdomains,
                           transition, n_points=50, tol_plot=1e-3):
    """
    2D 绘图: 传入具体 window_func(bump / cosine / sigmoid)，并绘制子域权重。
    """
    (x_lo, y_lo), (x_hi, y_hi) = domain_bounds
    x_vals = jnp.linspace(x_lo, x_hi, n_points)
    y_vals = jnp.linspace(y_lo, y_hi, n_points)
    X, Y = jnp.meshgrid(x_vals, y_vals, indexing="ij")
    XY_flat = jnp.column_stack([X.ravel(), Y.ravel()])

    xmins = jnp.stack([s[0] for s in subdomains])  # (n_sub,2)
    xmaxs = jnp.stack([s[1] for s in subdomains])
    wmins = jnp.full_like(xmins, transition)
    wmaxs = jnp.full_like(xmaxs, transition)

    # 计算原始权重
    w_raw = window_func(xmins, xmaxs, wmins, wmaxs, XY_flat, tol_plot)
    # 做归一
    denom = jnp.maximum(w_raw.sum(axis=1, keepdims=True), 1e-12)
    w_norm = w_raw / denom

    n_sub = len(subdomains)
    w_maps = [w_norm[:, i].reshape(n_points, n_points) for i in range(n_sub)]
    w_sum  = jnp.sum(w_norm, axis=1).reshape(n_points, n_points)

    fig, axs = plt.subplots(1, n_sub + 1,
                            figsize=(4.5*(n_sub + 1), 5),
                            subplot_kw=dict(aspect="equal"))
    if n_sub == 1:
        axs = [axs]

    fig.suptitle(f"2D Window Weights [{window_func.__name__}]", fontsize=15, y=1.02)

    # 每个子域的权重
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
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("weight", rotation=90)

    # 所有子域权重之和
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


##############################################################################
# 3) 主函数: 演示三种窗口函数在 1D / 2D 上的绘图
##############################################################################
if __name__ == "__main__":
    overlap = 0.2
    tol = 1e-8

    #-----------------------------------------------------------------
    # 1D 测试: 生成子域, 然后依次绘制 bump / cosine / sigmoid
    #-----------------------------------------------------------------
    domain_1d = (jnp.array([0.]), jnp.array([3.]))
    subs_1d = generate_subdomains(domain_1d, n_sub_per_dim=3, overlap=overlap)
    print(f"--- 1D 测试: {len(subs_1d)} 个子域 ---")

    # 1D - bump
    plot_window_weights_1d(bump_window_func, domain_1d, subs_1d, overlap,
                           n_points=400, tol_plot=tol)

    # 1D - cosine
    plot_window_weights_1d(cosine, domain_1d, subs_1d, overlap,
                           n_points=400, tol_plot=tol)

    # 1D - sigmoid
    plot_window_weights_1d(sigmoid, domain_1d, subs_1d, overlap,
                           n_points=400, tol_plot=tol)


    #-----------------------------------------------------------------
    # 2D 测试: 同样生成子域, 并绘制
    #-----------------------------------------------------------------
    domain_2d = (jnp.array([0., 0.]), jnp.array([1., 1.]))
    subs_2d = generate_subdomains(domain_2d, n_sub_per_dim=2, overlap=overlap)
    print(f"\n--- 2D 测试: {len(subs_2d)} 个子域 ---")

    # 2D - bump
    plot_window_weights_2d(bump_window_func, domain_2d, subs_2d, overlap,
                           n_points=80, tol_plot=tol)

    # 2D - cosine
    plot_window_weights_2d(cosine, domain_2d, subs_2d, overlap,
                           n_points=80, tol_plot=tol)

    # 2D - sigmoid
    plot_window_weights_2d(sigmoid, domain_2d, subs_2d, overlap,
                           n_points=80, tol_plot=tol)

    print("所有窗口函数测试完成！")
