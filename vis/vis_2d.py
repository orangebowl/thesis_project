import os
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np

def _ensure_dir(d):
    if d is not None:
        os.makedirs(d, exist_ok=True)

def plot_field_and_error(
    grid_x, grid_y,
    u_pred_grid, u_exact_grid,
    save_dir=None,
    title_prefix="FBPINN-2D",
    cmap="viridis",
    vmin=None, vmax=None,
):
    _ensure_dir(save_dir)
    err = jnp.abs(u_pred_grid - u_exact_grid)
    v = jnp.abs(err.max())
    fig, axs = plt.subplots(1, 3, figsize=(14, 4),
                            subplot_kw={"aspect": "equal"})
    extent = [grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]]
    for ax, data, title in zip(
        axs,
        [u_pred_grid, u_exact_grid, err],
        ["FBPINN Pred", "Exact", "Absolute Error"]
    ):
        im = ax.imshow(data, extent=extent, origin="lower",
                       cmap=cmap)
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
    plt.suptitle(f"{title_prefix}: field view", fontsize=14)
    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "field_and_error.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

def plot_training_curves(
    loss_hist,
    l1_steps=None, l1_hist=None,
    save_dir=None,
    title_prefix="2D",
):
    _ensure_dir(save_dir)
    step_axis = jnp.arange(len(loss_hist))
    plt.figure(figsize=(6, 4))
    plt.plot(step_axis, loss_hist, label="PDE Loss")
    if l1_steps is not None and l1_hist is not None and len(l1_steps):
        plt.plot(l1_steps, l1_hist, label="L1 Error")
    plt.yscale("log")
    plt.xlabel("Step"); plt.ylabel("Value")
    plt.title(f"{title_prefix}: training curves")
    plt.grid(ls=":"); plt.legend()
    if save_dir:
        path = os.path.join(save_dir, "training_curves.png")
        plt.savefig(path, dpi=150)
        plt.close()
        return path

def save_training_stats(loss_hist, l1_steps, l1_hist, final_metrics, save_dir):
    _ensure_dir(save_dir)
    save_dict = {
        "loss_hist": loss_hist,
        "l1_steps": l1_steps,
        "l1_hist": l1_hist,
        **final_metrics
    }
    jnp.savez(os.path.join(save_dir, "stats.npz"), **save_dict)
    print("[vis_2d] Final Metrics:")
    print(f"    - Relative L2 Error: {final_metrics['relative_l2_error']:.3e}")
    print(f"    - MSE: {final_metrics['mse']:.3e}")
    print(f"    - RMSE: {final_metrics['rmse']:.3e}")
    print(f"[vis_2d] stats saved to {save_dir}/stats.npz")

def visualize_2d(model,
                 grid_x, grid_y,
                 u_pred_grid, u_exact_grid,
                 loss_hist, l1_steps, l1_hist,
                 save_dir,
                 title_prefix="2D",
                 ):
    error = u_pred_grid - u_exact_grid
    rel_l2_error = jnp.linalg.norm(error) / jnp.linalg.norm(u_exact_grid)
    mse = jnp.mean(error**2)
    rmse = jnp.sqrt(mse)
    final_metrics = {
        "relative_l2_error": rel_l2_error,
        "mse": mse,
        "rmse": rmse,
    }
    paths = dict()
    paths["field_and_error"] = plot_field_and_error(
        grid_x, grid_y, u_pred_grid, u_exact_grid,
        save_dir, title_prefix
    )
    paths["training"] = plot_training_curves(
        loss_hist, l1_steps, l1_hist,
        save_dir, title_prefix
    )
    if hasattr(model, "subnets") and hasattr(model, "window_fn"):
        try:
            paths["window_weights"] = plot_window_weights(
                model, grid_x, grid_y,
                save_dir, title_prefix
            )
        except Exception as e:
            print(f"[vis_2d] Warning: Failed to plot window weights: {e}")
    save_training_stats(loss_hist, l1_steps, l1_hist, final_metrics, save_dir)
    return paths

def plot_window_weights(
    model,
    grid_x, grid_y,
    save_dir=None,
    title_prefix="FBPINN-2D",
    cmap="viridis",
):
    _ensure_dir(save_dir)
    Nx, Ny = len(grid_x), len(grid_y)
    mesh   = jnp.stack(jnp.meshgrid(grid_x, grid_y, indexing="ij"), -1)
    pts    = mesh.reshape(-1, model.xdim)
    if model.window_fn is None:
        from utils.window_function import cosine
        w = cosine(
            model.xmins_all, model.xmaxs_all,
            model.wmins_fixed, model.wmaxs_fixed,
            pts
        )
    else:
        w = model.window_fn(pts)
    w = w.T.reshape(len(model.subnets), Nx, Ny)
    ns = w.shape[0]
    ncols = min(4, ns)
    nrows = int(jnp.ceil(ns / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows),
                            subplot_kw={"aspect": "equal"})
    axs = axs.ravel()
    extent = [grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]]
    for i in range(ns):
        im = axs[i].imshow(w[i], extent=extent, origin="lower", cmap=cmap)
        axs[i].set_title(f"w[{i}]")
        plt.colorbar(im, ax=axs[i], fraction=0.046)
    for ax in axs[ns:]:
        ax.axis("off")
    plt.suptitle(f"{title_prefix}: window weights", fontsize=14)
    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "window_weights.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

def plot_subdomains(subdomains_list, *, save_path=None, cmap_name="tab20"):
    y_centers = np.array(
        [float((sd[0, 1] + sd[1, 1]) / 2) for sd in subdomains_list], dtype=float
    )
    unique_rows = np.unique(np.round(y_centers, 6))
    n_rows = len(unique_rows)
    cmap = plt.get_cmap(cmap_name, n_rows)
    row_color = {row: cmap(i) for i, row in enumerate(unique_rows)}
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect("equal")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"{len(subdomains_list)} Sub-domains ({n_rows} rows)")
    for sd, y_c in zip(subdomains_list, y_centers):
        (xL, yL), (xR, yR) = sd
        xL, yL, xR, yR = map(float, [xL, yL, xR, yR])
        color = row_color[float(np.round(y_c, 6))]
        rect = plt.Rectangle((xL, yL), xR - xL, yR - yL,
                             fc=color, ec="black", linewidth=1.2, alpha=0.35)
        ax.add_patch(rect)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Sub-domain plot saved to {save_path}")
    else:
        plt.show()

__all__ = [
    "plot_field_and_error",
    "plot_training_curves",
    "save_training_stats",
    "visualize_2d",
]