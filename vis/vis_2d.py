"""
utils.vis_2d
============

2-D FBPINN 可视化工具函数。

主要入口
--------
    visualize_2d(...)        # 一次生成“预测/真解/误差”+“训练曲线”并保存
"""

import os
import matplotlib.pyplot as plt
import jax.numpy as jnp


# -------------------------------------------------
# 通用小工具
# -------------------------------------------------
def _ensure_dir(d):
    if d is not None:
        os.makedirs(d, exist_ok=True)


# -------------------------------------------------
# (A) 预测、真解、绝对误差
# -------------------------------------------------
def plot_field_and_error(
    grid_x, grid_y,
    u_pred_grid, u_exact_grid,
    save_dir=None,
    title_prefix="FBPINN-2D",
    cmap="viridis",
    vmin=None, vmax=None,
):
    """
    绘制三张图：Pred / Exact / |Pred-Exact|
    参数
    ----
    grid_x, grid_y  : 1-D 网格坐标 (length = Nx, Ny)
    u_pred_grid     : (Nx, Ny)
    u_exact_grid    : (Nx, Ny)
    返回
    ----
    path(str) | None
    """
    _ensure_dir(save_dir)

    err_abs = jnp.abs(u_pred_grid - u_exact_grid)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4),
                            subplot_kw={"aspect": "equal"})
    extent = [grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]]

    for ax, data, title in zip(
        axs,
        [u_pred_grid, u_exact_grid, err_abs],
        ["FBPINN Pred", "Exact", "Absolute Error"]
    ):
        im = ax.imshow(data, extent=extent, origin="lower",
                       cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax)
        ax.set_title(title)

    plt.suptitle(f"{title_prefix}: field view", fontsize=14)
    plt.tight_layout()

    if save_dir:
        path = os.path.join(save_dir, "field_and_error.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path


# -------------------------------------------------
# (B) 训练损失 / L1 曲线
# -------------------------------------------------
def plot_training_curves(
    loss_hist,
    l1_steps=None, l1_hist=None,
    save_dir=None,
    title_prefix="FBPINN-2D",
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


# -------------------------------------------------
# (C) 保存训练统计
# -------------------------------------------------
def save_training_stats(loss_hist, l1_steps, l1_hist, save_dir):
    _ensure_dir(save_dir)
    jnp.savez(os.path.join(save_dir, "stats.npz"),
              loss_hist=loss_hist,
              l1_steps=l1_steps,
              l1_hist=l1_hist)
    print(f"[vis_2d] stats saved to {save_dir}/stats.npz")


# -------------------------------------------------
# (D) 一键可视化
# -------------------------------------------------
def visualize_2d(
    grid_x, grid_y,
    u_pred_grid, u_exact_grid,
    loss_hist, l1_steps, l1_hist,
    save_dir,
    title_prefix="FBPINN-2D",
):
    """聚合调用上述三个函数并返回 {标签: 路径} 字典"""
    paths = dict()
    paths["field_and_error"] = plot_field_and_error(
        grid_x, grid_y, u_pred_grid, u_exact_grid,
        save_dir, title_prefix
    )
    paths["training"] = plot_training_curves(
        loss_hist, l1_steps, l1_hist,
        save_dir, title_prefix
    )
    save_training_stats(loss_hist, l1_steps, l1_hist, save_dir)
    return paths


# -------------------------------------------------
# 导出符号
# -------------------------------------------------
__all__ = [
    "plot_field_and_error",
    "plot_training_curves",
    "save_training_stats",
    "visualize_2d",
]
