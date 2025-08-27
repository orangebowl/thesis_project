from __future__ import annotations
import os
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from pathlib import Path

def _ensure_dir(d):
    if d is not None:
        Path(d).mkdir(parents=True, exist_ok=True)

def plot_field_and_error(
    grid_x, grid_y,
    u_pred_grid, u_exact_grid,
    save_dir=None,
    title_prefix="PINN-2D",
    cmap="viridis",
):
    _ensure_dir(save_dir)
    difference = u_pred_grid - u_exact_grid
    
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), subplot_kw={"aspect": "equal"})
    extent = [grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]]
    
    # Unified colorbar range
    vmax_pred_exact = max(np.max(u_pred_grid), np.max(u_exact_grid))
    vmin_pred_exact = min(np.min(u_pred_grid), np.min(u_exact_grid))
    vmax_diff = np.max(np.abs(difference))
    
    im1 = axs[0].imshow(u_pred_grid, extent=extent, origin="lower", cmap=cmap, vmin=vmin_pred_exact, vmax=vmax_pred_exact)
    plt.colorbar(im1, ax=axs[0])
    axs[0].set_title("PINN Prediction")

    im2 = axs[1].imshow(u_exact_grid, extent=extent, origin="lower", cmap=cmap, vmin=vmin_pred_exact, vmax=vmax_pred_exact)
    plt.colorbar(im2, ax=axs[1])
    axs[1].set_title("Exact Solution")

    im3 = axs[2].imshow(difference, extent=extent, origin="lower", cmap=cmap, vmin=-vmax_diff, vmax=vmax_diff)
    plt.colorbar(im3, ax=axs[2])
    axs[2].set_title("Difference")
    
    plt.suptitle(f"{title_prefix}: Field Comparison", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_dir:
        path = Path(save_dir) / "field_and_error.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

def plot_training_curves(
    loss_hist,
    rel_l2_steps=None, 
    rel_l2_hist=None,
    save_dir=None,
    title_prefix="2D",
):
    _ensure_dir(save_dir)
    step_axis = np.arange(len(loss_hist))
    plt.figure(figsize=(6, 4))
    plt.plot(step_axis, loss_hist, label="PDE Loss", alpha=0.8)
    if rel_l2_steps is not None and rel_l2_hist is not None and len(rel_l2_steps) > 0:
        plt.plot(rel_l2_steps, rel_l2_hist, label="Relative L2 Error",alpha=0.8)
        
    plt.yscale("log")
    plt.xlabel("Training Step"); plt.ylabel("Value (log scale)")
    plt.title(f"{title_prefix}: Training Curves")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    if save_dir:
        path = Path(save_dir) / "training_curves.png"
        plt.savefig(path, dpi=150)
        plt.close()
        return path

def save_training_stats(loss_hist, rel_l2_steps, rel_l2_hist, final_metrics, save_dir):
    _ensure_dir(save_dir)
    save_dict = {
        "loss_hist": np.array(loss_hist),
        "eval_steps": np.array(rel_l2_steps),
        "rel_l2_hist": np.array(rel_l2_hist),
        **{k: np.array(v) for k, v in final_metrics.items()}
    }
    np.savez(Path(save_dir) / "stats.npz", **save_dict)
    print("\n[Visualizer] Final Metrics:")
    print(f"  - Relative L2 Error: {final_metrics['relative_l2_error']:.4e}")
    # MODIFIED: Added MAE to the print output
    print(f"  - MAE              : {final_metrics['mae']:.4e}")
    print(f"  - MSE              : {final_metrics['mse']:.4e}")
    print(f"  - RMSE             : {final_metrics['rmse']:.4e}")
    print(f"[Visualizer] Stats saved to {Path(save_dir) / 'stats.npz'}")

def visualize_2d(
    model,
    grid_x, grid_y,
    u_pred_grid, u_exact_grid,
    loss_hist, rel_l2_steps, rel_l2_hist,
    save_dir,
    title_prefix="2D",
):
    error = u_pred_grid - u_exact_grid
    
    # MODIFIED: Added MAE calculation
    final_metrics = {
        "relative_l2_error": jnp.linalg.norm(error) / (jnp.linalg.norm(u_exact_grid) + 1e-8),
        "mae": jnp.mean(jnp.abs(error)),
        "mse": jnp.mean(error**2),
        "rmse": jnp.sqrt(jnp.mean(error**2)),
    }
    
    paths = {}
    paths["field_and_error"] = plot_field_and_error(
        grid_x, grid_y, u_pred_grid, u_exact_grid,
        save_dir, title_prefix
    )
    paths["training_curves"] = plot_training_curves(
        loss_hist, rel_l2_steps, rel_l2_hist,
        save_dir, title_prefix
    )
    
    save_training_stats(loss_hist, rel_l2_steps, rel_l2_hist, final_metrics, save_dir)
    
    return paths, final_metrics


import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Iterable, Optional, Tuple

def _to_np(a):
    return np.asarray(a) if not isinstance(a, np.ndarray) else a

def plot_subdomains(
    sub_arr: Iterable, 
    *,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = "FBPINN Subdomains (2D)",
    show_index: bool = False,
    show_centers: bool = True,
    fill_alpha: float = 0.15,
    edge_alpha: float = 0.9,
    linewidth: float = 1.2,
    dpi: int = 200,
    domain: Optional[Tuple[np.ndarray, np.ndarray]] = None,  # (lo, hi) for frame
):
    """
    可视化 2D 子域：
    - sub_arr: 可迭代对象，元素为 (2,2) 的数组或元组 [left, right]，left/right = (x, y)
               与你调用中 `jnp.stack(sd, 0)` 的结果一致。
    - domain:  可选 (lo, hi)，用于画外框；若不提供，则根据子域自适应边界。
    """
    # 1) 规范化输入：转为 np，并检查形状
    rects = []
    for sd in sub_arr:
        sd_np = _to_np(sd)
        # 也兼容传进来还是 (left, right) 元组的情况
        if sd_np.ndim == 1 and len(sd_np) == 2 and np.asarray(sd_np[0]).shape == (2,):
            sd_np = np.stack(sd_np, axis=0)  # -> (2,2)
        assert sd_np.shape == (2, 2), f"Each subdomain must be (2,2) [left,right], got {sd_np.shape}"
        left, right = sd_np[0], sd_np[1]
        lx, ly = float(left[0]), float(left[1])
        rx, ry = float(right[0]), float(right[1])
        w, h = (rx - lx), (ry - ly)
        if w <= 0 or h <= 0:
            # 跳过异常子域（可能是极端裁剪后宽/高为0）
            continue
        rects.append((lx, ly, w, h))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    else:
        fig = ax.figure

    # 2) 设置边界与外框
    if len(rects) == 0:
        ax.set_title("No subdomains to plot.")
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        return fig, ax

    if domain is not None:
        d_lo, d_hi = map(_to_np, domain)
        xmin, ymin = float(d_lo[0]), float(d_lo[1])
        xmax, ymax = float(d_hi[0]), float(d_hi[1])
    else:
        xs = [lx for lx, ly, w, h in rects] + [lx + w for lx, ly, w, h in rects]
        ys = [ly for lx, ly, w, h in rects] + [ly + h for lx, ly, w, h in rects]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        # 适当留白
        pad_x = 0.02 * (xmax - xmin + 1e-12)
        pad_y = 0.02 * (ymax - ymin + 1e-12)
        xmin, xmax = xmin - pad_x, xmax + pad_x
        ymin, ymax = ymin - pad_y, ymax + pad_y

    # 外框（可选）
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                           fill=False, linewidth=1.0, linestyle='-', alpha=0.6))

    # 3) 配色：使用 tab20 循环
    cmap = plt.cm.get_cmap("tab20")
    n_colors = cmap.N

    # 4) 绘制子域
    for i, (lx, ly, w, h) in enumerate(rects):
        color = cmap(i % n_colors)
        ax.add_patch(Rectangle((lx, ly), w, h,
                               facecolor=color, edgecolor=color,
                               linewidth=linewidth, alpha=fill_alpha))
        ax.add_patch(Rectangle((lx, ly), w, h,
                               fill=False, edgecolor=color,
                               linewidth=linewidth, alpha=edge_alpha))
        if show_centers or show_index:
            cx, cy = lx + 0.5 * w, ly + 0.5 * h
            if show_centers:
                ax.plot([cx], [cy], marker='o', markersize=2.8, alpha=0.9)
            if show_index:
                ax.text(cx, cy, str(i), ha="center", va="center", fontsize=8)

    # 5) 轴样式
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    ax.grid(False)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)

    return fig, ax