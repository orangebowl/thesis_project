### in this file we can choose three kind of window functions: bump cosine and sigmoid
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils.data_utils import generate_subdomains



def bump_window_func(xmins_all, xmaxs_all, wmins_all, wmaxs_all, x, eps=1e-8):
    x = jnp.atleast_2d(x)  # (N,d)
    N, d = x.shape
    n_sub = xmins_all.shape[0]

    x_   = x[:, None, :]              
    xmin = xmins_all[None, :, :]
    xmax = xmaxs_all[None, :, :]

    mu = (xmin + xmax) / 2.0
    sd = (xmax - xmin) / 2.0 + eps
    t  = (x_ - mu) / sd

    inside = jnp.abs(t) < 1.0
    core   = jnp.exp(3.0 / (t**2 - 1.001)) / jnp.exp(-3.0)
    w_dim  = jnp.where(inside, core, 0.0)
    w_raw = jnp.prod(w_dim, axis=-1)
    return w_raw

Pi = jnp.pi

def cosine(xmins_all, xmaxs_all,
                           wmins_all, wmaxs_all,     
                           x, tol: float = 1e-12):
    """
    Cos² window 
       input: xmins_all, xmaxs_all : (n_sub, d)
        x                    : (N,d) or (d,)
    return
        w : (N, n_sub)   already normalized,  ∑_sub w = 1
    """
    xmins_all = jnp.asarray(xmins_all)
    xmaxs_all = jnp.asarray(xmaxs_all)
    x         = jnp.asarray(x)

    # Ensure all boundary arrays are (n_sub, d)
    if xmins_all.ndim == 1:
        xmins_all = xmins_all[:, None]
        xmaxs_all = xmaxs_all[:, None]

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim == 0:
        x = x[None, None]

    x_   = x[:, None, :]                    # (N,1,d)
    xmin = xmins_all[None, :, :]            # (1,n_sub,d)
    xmax = xmaxs_all[None, :, :]

    mu = (xmin + xmax) / 2.0
    sd = (xmax - xmin) / 2.0 

    r     = (x_ - mu) / sd                 # normalized to [-1,1]
    r_clipped = jnp.clip(r, -1.0, 1.0)
    core = ((1.0 + jnp.cos(jnp.pi * r_clipped)))**2
    #w_dim = jnp.heaviside(x_ -xmin, 1)*jnp.heaviside(xmax - x_,1)* core 
    w_dim = core
    w_raw = jnp.prod(w_dim, axis=-1)        # (N, n_sub)
    denom = jnp.maximum(w_raw.sum(axis=1, keepdims=True), 1e-12)
    w = w_raw / denom                       # (N, n_sub)
    
    return w


def sigmoid(xmins_all, xmaxs_all,
            wmins_all, wmaxs_all,
            x, tol: float = 1e-8):
    xmins_all = jnp.asarray(xmins_all)
    xmaxs_all = jnp.asarray(xmaxs_all)
    wmins_all = jnp.asarray(wmins_all)
    wmaxs_all = jnp.asarray(wmaxs_all)
    x         = jnp.asarray(x)

    # Ensure all boundary arrays are (n_sub, d)
    if xmins_all.ndim == 1:
        xmins_all = xmins_all[:, None]
        xmaxs_all = xmaxs_all[:, None]
        wmins_all = wmins_all[:, None]
        wmaxs_all = wmaxs_all[:, None]

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim == 0:
        x = x[None, None]

    assert x.ndim == 2, f"x.ndim != 2 → got x.shape={x.shape}"
    assert xmins_all.ndim == 2, f"xmins_all shape wrong: {xmins_all.shape}"

    # Broadcast to (N, n_sub, d)
    x_    = x[:, None, :]                  # (N, 1, d)
    xmin  = xmins_all[None, :, :]          # (1, n_sub, d)
    xmax  = xmaxs_all[None, :, :]
    wmin  = wmins_all[None, :, :] + tol
    wmax  = wmaxs_all[None, :, :] + tol

    # Pair-sigmoid computation
    t = jnp.log((1 - tol) / tol)

    mu_min = xmin + wmin / 2.0
    sd_min = wmin / (2.0 * t)

    mu_max = xmax - wmax / 2.0
    sd_max = wmax / (2.0 * t)

    left   = jax.nn.sigmoid((x_ - mu_min) / sd_min)
    right  = jax.nn.sigmoid((mu_max - x_) / sd_max)

    w_dim = left * right                  # (N, n_sub, d)
    w_raw = jnp.prod(w_dim, axis=-1)     # (N, n_sub)
    
    denom = jnp.maximum(w_raw.sum(axis=1, keepdims=True), 1e-12)
    w = w_raw / denom                       # (N, n_sub)

    return w




def plot_window_weights_1d(window_func, domain_bounds, subdomains,
                           transition, n_points=300, tol_plot=1e-3):

    lo, hi = domain_bounds
    x_plot = jnp.linspace(lo[0], hi[0], n_points).reshape(-1, 1)

    xmins = jnp.stack([s[0] for s in subdomains])
    xmaxs = jnp.stack([s[1] for s in subdomains])
    wmins = jnp.full_like(xmins, transition)
    wmaxs = jnp.full_like(xmaxs, transition)

    w_raw = window_func(xmins, xmaxs, wmins, wmaxs, x_plot, tol_plot)
    denom = jnp.maximum(w_raw.sum(axis=1, keepdims=True), 1e-12)
    w = w_raw / denom

    fig = plt.figure(figsize=(10.5, 4.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[4.0, 0.7], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1], sharex=ax)

    color_cycle = plt.cm.tab10.colors

    for j in range(w.shape[1]):
        ax.plot(x_plot.squeeze(),
                w[:, j],
                label=f"Subdomain {j}",
                lw=2,
                color=color_cycle[j % len(color_cycle)])
    ax.plot(x_plot.squeeze(),
            jnp.sum(w, axis=1),
            "k--",
            lw=2,
            label="Sum of weights")

    ax.set_ylabel("weight", fontsize=12)
    ax.set_title(f"1D Window Weights [{window_func.__name__}]  (overlap={transition})",
                 fontsize=14, pad=10)

    ax.legend(loc="upper right", fontsize=9)
    ax.tick_params(axis="x", labelbottom=False) 
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(-0.02, 1.05)

    bars = []
    colors = []
    for j, (xmin, xmax) in enumerate(subdomains):
        bars.append((xmin[0], xmax[0] - xmin[0]))
        colors.append(color_cycle[j % len(color_cycle)])


    ax_bar.broken_barh(bars, (0, 1), facecolors=colors, alpha=0.35, edgecolors="none")

    for j, (xmin, xmax) in enumerate(subdomains):
        ax_bar.text((xmin[0] + xmax[0]) / 2, 0.5, f"{j}",
                    ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")

    ax_bar.set_xlim(lo[0], hi[0])


    plt.tight_layout()
    plt.show()


def plot_window_weights_2d(window_func, domain_bounds, subdomains,
                           transition, n_points=50, tol_plot=1e-3):

    (x_lo, y_lo), (x_hi, y_hi) = domain_bounds
    x_vals = jnp.linspace(x_lo, x_hi, n_points)
    y_vals = jnp.linspace(y_lo, y_hi, n_points)
    X, Y = jnp.meshgrid(x_vals, y_vals, indexing="ij")
    XY_flat = jnp.column_stack([X.ravel(), Y.ravel()])

    xmins = jnp.stack([s[0] for s in subdomains])  # (n_sub,2)
    xmaxs = jnp.stack([s[1] for s in subdomains])
    wmins = jnp.full_like(xmins, transition)
    wmaxs = jnp.full_like(xmaxs, transition)

    w_raw = window_func(xmins, xmaxs, wmins, wmaxs, XY_flat, tol_plot)

    denom = jnp.maximum(w_raw.sum(axis=1, keepdims=True), 1e-12)
    w_norm = w_raw / denom

    n_sub = len(subdomains)
    w_maps = [w_norm[:, i].reshape(n_points, n_points) for i in range(n_sub)]
    w_sum  = jnp.sum(w_norm, axis=1).reshape(n_points, n_points)

    fig, axs = plt.subplots(1, n_sub + 1,
                            figsize=(4.2*(n_sub + 1), 4.8),
                            subplot_kw=dict(aspect="equal"))
    if n_sub == 1:
        axs = [axs]

    fig.suptitle(f"2D Window Weights [{window_func.__name__}]", fontsize=16, y=1.04)

    norm_sub = mcolors.Normalize(vmin=0.0, vmax=1.0)
    for i, ax in enumerate(axs[:-1]):
        im = ax.imshow(
            w_maps[i],
            extent=[x_lo, x_hi, y_lo, y_hi],
            origin="lower",
            cmap="coolwarm",
            norm=norm_sub,
            interpolation="bilinear"
        )
        ax.set_title(f"Subdomain {i}", fontsize=13)
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("y", fontsize=11)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        cb.set_label("weight", rotation=90, fontsize=10)
        cb.ax.tick_params(labelsize=9)

    ax_sum = axs[-1]
    vmax_sum = float(w_sum.max())
    norm_sum = mcolors.Normalize(vmin=0.0, vmax=max(1.0, vmax_sum))
    im_sum = ax_sum.imshow(
        w_sum,
        extent=[x_lo, x_hi, y_lo, y_hi],
        origin="lower",
        cmap="viridis",
        norm=norm_sum,
        interpolation="bilinear"
    )
    ax_sum.set_title("Sum of Weights", fontsize=13)
    ax_sum.set_xlabel("x", fontsize=11)
    ax_sum.set_ylabel("y", fontsize=11)
    cb_sum = fig.colorbar(im_sum, ax=ax_sum, fraction=0.045, pad=0.04)
    cb_sum.set_label("sum of weights", rotation=90, fontsize=10)
    cb_sum.ax.tick_params(labelsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


if __name__ == "__main__":
    overlap = 0.1
    tol = 1e-8

    # 1d plot
    domain_1d = (jnp.array([0.]), jnp.array([1.]))
    subs_1d = generate_subdomains(domain_1d, n_sub_per_dim=4, overlap=overlap)
    print(f"1D test: {len(subs_1d)} ")

    # 1D - bump
    plot_window_weights_1d(bump_window_func, domain_1d, subs_1d, overlap,
                           n_points=400, tol_plot=tol)

    # 1D - cosine
    plot_window_weights_1d(cosine, domain_1d, subs_1d, overlap,
                           n_points=400, tol_plot=tol)

    # 1D - sigmoid
    plot_window_weights_1d(sigmoid, domain_1d, subs_1d, overlap,
                           n_points=400, tol_plot=tol)


    # 2D
    domain_2d = (jnp.array([0., 0.]), jnp.array([1., 1.]))
    subs_2d = generate_subdomains(domain_2d, n_sub_per_dim=2, overlap=overlap)
    print(f"\n 2D test: {len(subs_2d)} ")

    # 2D - bump
    plot_window_weights_2d(bump_window_func, domain_2d, subs_2d, overlap,
                           n_points=80, tol_plot=tol)

    # 2D - cosine
    plot_window_weights_2d(cosine, domain_2d, subs_2d, overlap,
                           n_points=80, tol_plot=tol)

    # 2D - sigmoid
    plot_window_weights_2d(sigmoid, domain_2d, subs_2d, overlap,
                           n_points=80, tol_plot=tol)

