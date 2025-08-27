"""
utils.vis_1d
============

集中放置 **所有** 1-D FBPINN（或 PINN） 画图/保存 工具函数。并提供一个
`visualize_rel_l2_error` 函数作为统一入口，只需调用这一个函数即可完成所有可视化。

用法
----
    import utils.vis_1d as vis

    vis.visualize_rel_l2_error(
        x_test.squeeze(),   # (N,)
        u_true,             # (N,)
        u_pred,             # (N,)
        loss_hist,          # (steps,)
        rel_l2_steps,       # (num_check,)
        rel_l2_hist,        # (num_check,)
        subdomains={"model": model, "list": subdomains_list},
        overlap=overlap,
        save_dir=SAVE_DIR,
    )

如果只想画 PINN 的结果（没有子域），则传 `subdomains=None` 即可跳过子域/窗口相关的绘图。
"""

import os
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from utils.window_function import cosine as _cosine


# -------------------------------------------------
# 内部通用小工具
# -------------------------------------------------
def _ensure_dir(d):
    if d is not None:
        os.makedirs(d, exist_ok=True)


def plot_subdomain_partials(model, x_test, u_true, save_dir=None):
    _ensure_dir(save_dir)


    x_in = jnp.expand_dims(x_test, axis=1)  # (N,1)

    ns = len(model.subnets)
    parts_w_ansatz = []

    # _cosine(xmins, xmaxs, wmins, wmaxs, x_batch)
    w_all = _cosine(
        model.xmins_all,         # (ns,1)
        model.xmaxs_all,         # (ns,1)
        model.wmins_all_fixed,   # (ns,1)
        model.wmaxs_all_fixed,   # (ns,1)
        x_in                     # (N,1)
    )  # (N, ns)

    for i in range(ns):
        # 1) 归一化到子域 i：xnorm = (x_in - center_i) / scale_i
        left_i   = model.xmins_all[i]
        right_i  = model.xmaxs_all[i]
        center_i = (left_i + right_i) / 2.0
        scale_i  = (right_i - left_i) / 2.0
        xnorm_i  = (x_in - center_i) / jnp.maximum(scale_i, 1e-9)  # (N,1)

        # 2) 子网原始输出
        raw_i = model.subnets[i](xnorm_i).squeeze(axis=1)  # (N,)

        # 3) 窗口加权 + ansatz
        part_i = model.ansatz(x_in, (w_all[:, i] * raw_i)[:, None]).squeeze()  # (N,)
        parts_w_ansatz.append(part_i)

    parts_w_ansatz = jnp.stack(parts_w_ansatz, axis=1)  # (N, ns)

    # 绘图：左轴画各子域贡献与真解；右轴叠加各 window function
    fig, ax1 = plt.subplots(figsize=(10, 5))
    for i in range(ns):
        ax1.plot(x_test, parts_w_ansatz[:, i], label=f"sub {i}")
    ax1.plot(x_test, u_true, "--", lw=2, label="exact")

    ax1.set_xlabel("x"); ax1.set_ylabel("u")
    ax1.set_title("Subdomain contributions (left) + Window functions (right)")
    ax1.grid(ls=":")
    leg1 = ax1.legend(fontsize=8, loc="upper left")

    # 右轴：window functions，范围 0~1
    ax2 = ax1.twinx()
    for i in range(ns):
        ax2.plot(x_test, w_all[:, i], alpha=0.6)  # 不额外加标签，避免 legend 过长
    sum_w = jnp.sum(w_all, axis=1)
    ax2.plot(x_test, sum_w, lw=2)  # sum of windows
    ax2.set_ylabel("window weight")
    ax2.set_ylim(-0.05, 1.05)

    # 单独给 sum(w_i) 做个图例条目
    lines_sum = ax2.get_lines()[-1]
    leg2 = ax2.legend([lines_sum], ["sum w_i(x)"], loc="upper right", fontsize=8)

    # 让两个 legend 都显示
    ax1.add_artist(leg1)
    ax2.add_artist(leg2)

    if save_dir:
        path = os.path.join(save_dir, "subdomain_partials_with_windows.png")
        plt.savefig(path, dpi=150); plt.close()
        return path


# -------------------------------------------------
# 2) 预测 vs 真解
# -------------------------------------------------
def plot_prediction_vs_exact(x, u_true, u_pred, save_dir=None):
    _ensure_dir(save_dir)

    plt.figure()
    plt.plot(x, u_pred, label="Predicted")
    plt.plot(x, u_true, "--", label="Exact")
    plt.title("Prediction vs Exact"); plt.grid(ls=":"); plt.legend()

    if save_dir:
        path = os.path.join(save_dir, "prediction_vs_exact.png")
        plt.savefig(path, dpi=150); plt.close()
        return path


# -------------------------------------------------
# 3) 训练曲线
# -------------------------------------------------
def plot_training_loss(loss_hist, save_dir=None):
    _ensure_dir(save_dir)

    plt.figure()
    plt.plot(loss_hist, label="PDE Loss")
    plt.yscale("log"); plt.xlabel("step")
    plt.title("Training loss"); plt.grid(ls=":"); plt.legend()

    if save_dir:
        path = os.path.join(save_dir, "loss_curve.png")
        plt.savefig(path, dpi=150); plt.close()
        return path


def plot_test_rel_l2_curve(test_steps, rel_l2_hist, save_dir=None):
    _ensure_dir(save_dir)

    plt.figure()
    plt.plot(test_steps, rel_l2_hist, label="Relative L2 Error")
    plt.yscale("log"); plt.xlabel("step")
    plt.title("Test Relative L2"); plt.grid(ls=":"); plt.legend()

    if save_dir:
        path = os.path.join(save_dir, "rel_l2_curve.png")
        plt.savefig(path, dpi=150); plt.close()
        return path


# -------------------------------------------------
# 4) POU / window 函数
# -------------------------------------------------
def plot_window_weights(x, subdomains_list, overlap, save_dir=None):
    _ensure_dir(save_dir)

    xmins = jnp.stack([s[0] for s in subdomains_list])  # (ns,1)
    xmaxs = jnp.stack([s[1] for s in subdomains_list])  # (ns,1)
    wmins = jnp.full((len(subdomains_list), 1), overlap)
    wmaxs = jnp.full((len(subdomains_list), 1), overlap)

    x_in = x[:, None]  # (N,1)
    w_all = _cosine(xmins, xmaxs, wmins, wmaxs, x_in)  # (N, ns)

    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(w_all.shape[1]):
        ax.plot(x, w_all[:, i])
    ax.set_title("Window weights (from list + overlap)")
    ax.grid(ls=":")

    y_bottom = -0.2; height = 0.08
    for (xmin, xmax) in zip(xmins, xmaxs):
        xmin = float(xmin[0]); xmax = float(xmax[0])
        width = xmax - xmin; ov = width * float(overlap)
        ax.add_patch(plt.Rectangle((xmin + ov, y_bottom), width - 2 * ov, height,
                                   facecolor='lightgray', edgecolor='k', alpha=0.5))
        ax.add_patch(plt.Rectangle((xmin, y_bottom), ov, height,
                                   facecolor='salmon', edgecolor='none', alpha=0.5))
        ax.add_patch(plt.Rectangle((xmax - ov, y_bottom), ov, height,
                                   facecolor='salmon', edgecolor='none', alpha=0.5))
    ax.text(x[0], y_bottom - 0.07, 'gray = subdomain, red = overlap', fontsize=8)
    ax.set_ylim(y_bottom - 0.15, 1.05)

    if save_dir:
        path = os.path.join(save_dir, "window_weights.png")
        plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
        return path


def plot_window_functions_from_model(model, x, save_dir=None):
    """
    使用模型内部的 xmins/xmaxs/wmins/wmaxs 计算实际训练时的 window functions，
    并绘制所有 w_i(x) 及 sum_i w_i(x)。
    """
    _ensure_dir(save_dir)
    x_in = x[:, None]
    w_all = _cosine(
        model.xmins_all, model.xmaxs_all,
        model.wmins_all_fixed, model.wmaxs_all_fixed,
        x_in
    )  # (N, ns)
    sum_w = jnp.sum(w_all, axis=1)

    plt.figure(figsize=(10, 4))
    for i in range(w_all.shape[1]):
        plt.plot(x, w_all[:, i], alpha=0.8)
    plt.plot(x, sum_w, lw=2, label="sum w_i(x)")
    plt.ylim(-0.05, 1.05)
    plt.title("Window functions (from model)")
    plt.grid(ls=":")
    plt.legend()

    if save_dir:
        path = os.path.join(save_dir, "window_functions_model.png")
        plt.savefig(path, dpi=150); plt.close()
        return path


# -------------------------------------------------
# 5) 训练统计保存
# -------------------------------------------------
def save_training_stats(loss_hist, test_steps, rel_l2_hist, save_dir):
    _ensure_dir(save_dir)
    jnp.savez(os.path.join(save_dir, "stats.npz"),
              loss_hist=loss_hist,
              rel_l2_steps=test_steps,
              rel_l2_hist=rel_l2_hist)
    print(f"[vis_1d] stats saved to {save_dir}/stats.npz")


# -------------------------------------------------
# 6) 统一入口（Relative L2 版本）
# -------------------------------------------------
def visualize_rel_l2_error(
    x_test:    jnp.ndarray,
    u_true:    jnp.ndarray,
    u_pred:    jnp.ndarray,
    loss_hist: jnp.ndarray,
    rel_l2_steps:  jnp.ndarray,
    rel_l2_hist:   jnp.ndarray,
    subdomains: dict = None,     # {"model": FBPINN, "list": subdomains_list}
    overlap:   float = None,
    save_dir:  str = None,
):
    """
    一次性完成 1D 全部可视化：
      - Prediction vs Exact
      - Training Loss
      - Test Relative L2 曲线
      - （可选）Subdomain contributions + 右轴 Window functions
      - （可选）基于 list+overlap 的 Window weights（示意）
      - （可选）基于 model 参数的 Window functions（含 sum w_i）
    """
    _ensure_dir(save_dir)
    paths = {}

    # (1) Prediction vs Exact
    paths["pred_vs_exact"] = plot_prediction_vs_exact(x_test, u_true, u_pred, save_dir)

    # (2) Training loss
    paths["train_loss"] = plot_training_loss(loss_hist, save_dir)

    # (3) Test Relative L2 curve
    paths["test_rel_l2"] = plot_test_rel_l2_curve(rel_l2_steps, rel_l2_hist, save_dir)

    # (4) 子域与窗口
    if subdomains is not None:
        fb_model        = subdomains["model"]
        subdomains_list = subdomains["list"]

        # (4.1) 子域贡献 + 右轴窗口
        if hasattr(fb_model, 'subnets'):
            paths["subdomain_partials_with_windows"] = plot_subdomain_partials(
                model=fb_model, x_test=x_test, u_true=u_true, save_dir=save_dir
            )

        # (4.2) 基于 list+overlap 的窗口示意
        paths["window_weights"] = plot_window_weights(
            x=x_test, subdomains_list=subdomains_list, overlap=overlap, save_dir=save_dir
        )

        # (4.3) 基于 model 参数计算的真实窗口函数（含 sum）
        paths["window_functions_model"] = plot_window_functions_from_model(
            fb_model, x_test, save_dir
        )

    # (5) 保存训练统计
    paths["stats"] = save_training_stats(loss_hist, rel_l2_steps, rel_l2_hist, save_dir)

    return paths


# -------------------------------------------------
# 7) 兼容包装：保留 visualize_1d（把 l1_* 当作 rel_l2_* 用）
# -------------------------------------------------
def visualize_1d(
    x_test:    jnp.ndarray,
    u_true:    jnp.ndarray,
    u_pred:    jnp.ndarray,
    loss_hist: jnp.ndarray,
    l1_steps:  jnp.ndarray,     # 兼容旧参数名
    l1_hist:   jnp.ndarray,     # 兼容旧参数名
    subdomains: dict = None,
    overlap:   float = None,
    save_dir:  str = None,
):
    return visualize_rel_l2_error(
        x_test=x_test, u_true=u_true, u_pred=u_pred,
        loss_hist=loss_hist,
        rel_l2_steps=l1_steps, rel_l2_hist=l1_hist,
        subdomains=subdomains, overlap=overlap, save_dir=save_dir
    )


# -------------------------------------------------
# 8) 导出符号
# -------------------------------------------------
__all__ = [
    "plot_subdomain_partials",
    "plot_prediction_vs_exact",
    "plot_training_loss",
    "plot_test_rel_l2_curve",
    "plot_window_weights",
    "plot_window_functions_from_model",
    "save_training_stats",
    "visualize_rel_l2_error",
    "visualize_1d",
]
