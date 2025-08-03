"""
utils.vis_1d
============

集中放置 **所有** 1-D FBPINN（或 PINN） 画图/保存 工具函数。并提供一个
`visualize_1d` 函数作为统一入口，只需调用这一个函数即可完成所有可视化。

用法
----
    import utils.vis_1d as vis

    # 训练完成后，用：
    vis.visualize_1d(
        x_test.squeeze(),   # 形状 (N,)
        u_true,             # 形状 (N,)
        u_pred,             # 形状 (N,)
        loss_hist,          # 形状 (steps,)
        l1_steps,           # 形状 (num_check,)
        l1_hist,            # 形状 (num_check,)
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

# -------------------------------------------------
# 1. 子域贡献可视化（仅 FBPINN 时使用）
# -------------------------------------------------
def plot_subdomain_partials(model, x_test, u_true, save_dir=None):
    """
    对于 FBPINN，每个子网的“原始输出 × 窗口权重 × ansatz”与真解对比。
    model: FBPINN 对象
    x_test: 一维坐标，shape (N,)
    u_true: 真解，shape (N,)
    save_dir: 保存目录
    """
    _ensure_dir(save_dir)

    # 将 x_test 转成 (N,1) 形式供子网前向与窗口函数计算
    x_in = jnp.expand_dims(x_test, axis=1)  # (N,1)

    ns = len(model.subnets)
    parts_w_ansatz = []

    # 先一次性计算所有窗口权重：w_all.shape = (N, ns)
    # _cosine 的签名： (xmins_all, xmaxs_all, wmins_all, wmaxs_all, x_batch)
    w_all = _cosine(
        model.xmins_all,      # (ns,1)
        model.xmaxs_all,      # (ns,1)
        model.wmins_all_fixed,    # (ns,1)
        model.wmaxs_all_fixed,    # (ns,1)
        x_in                  # (N,1)
    )  # 返回 (N, ns)

    for i in range(ns):
        # 1) 归一化到子域 i：xnorm = (x_in - center_i) / scale_i
        left_i  = model.xmins_all[i]   # shape (1,)
        right_i = model.xmaxs_all[i]   # shape (1,)
        center_i = (left_i + right_i) / 2.0
        scale_i  = (right_i - left_i) / 2.0
        # x_in shape (N,1), center_i 标量或 (1,)
        xnorm_i = (x_in - center_i) / jnp.maximum(scale_i, 1e-9)  # (N,1)

        # 2) 子网原始输出：FCN 输入 xnorm_i，得到 (N,1)，然后 squeeze 为 (N,)
        raw_i = model.subnets[i](xnorm_i)          # (N,1)
        raw_i = raw_i.squeeze(axis=1)             # (N,)

        # 3) 对应窗口权重 w_i：从 w_all 中取第 i 列 (N,)
        w_i = w_all[:, i]                         # (N,)

        # 4) 窗口加权 + ansatz：先做 w_i * raw_i，然后再交给 ansatz
        part_i = model.ansatz(x_in, (w_i * raw_i)[:, None])  # (N,1) or (N,)
        parts_w_ansatz.append(part_i.squeeze())             # (N,)

    # 把所有子域贡献叠一个维度：stack → shape (N, ns)
    parts_w_ansatz = jnp.stack(parts_w_ansatz, axis=1)

    # 绘图
    plt.figure(figsize=(10, 5))
    for i in range(ns):
        plt.plot(x_test, parts_w_ansatz[:, i], label=f"sub {i}")
    plt.plot(x_test, u_true, "k--", lw=2, label="exact")

    plt.xlabel("x"); plt.ylabel("u")
    plt.title("Subdomain contributions")
    plt.legend(fontsize=8); plt.grid(ls=":")

    if save_dir:
        path = os.path.join(save_dir, "subdomain_partials.png")
        plt.savefig(path, dpi=150); plt.close()
        return path

# -------------------------------------------------
# 2. 预测 vs 真解
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
# 3. 训练曲线
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

def plot_test_l1_curve(test_steps, test_l1, save_dir=None):
    _ensure_dir(save_dir)

    plt.figure()
    plt.plot(test_steps, test_l1, label="L1 Error")
    plt.yscale("log"); plt.xlabel("step")
    plt.title("Test L1"); plt.grid(ls=":"); plt.legend()

    if save_dir:
        path = os.path.join(save_dir, "l1_curve.png")
        plt.savefig(path, dpi=150); plt.close()
        return path

# -------------------------------------------------
# 4. POU / window 函数
# -------------------------------------------------
def plot_window_weights(x, subdomains_list, overlap, save_dir=None):
    """
    在 1D 下绘制所有子域的窗口函数权重，并在底部标出子域和 overlap 区域。
    x: 一维坐标 (N,)
    subdomains_list: list of (left, right)，每个是 shape (1,) 的数组
    overlap: float，子域 overlap ratio（用来生成 wmins, wmaxs）
    """
    _ensure_dir(save_dir)

    xmins = jnp.stack([s[0] for s in subdomains_list])  # (ns,1)
    xmaxs = jnp.stack([s[1] for s in subdomains_list])  # (ns,1)
    wmins = jnp.full((len(subdomains_list), 1), overlap)
    wmaxs = jnp.full((len(subdomains_list), 1), overlap)

    x_in = x[:, None]  # (N,1)
    w_all = _cosine(xmins, xmaxs, wmins, wmaxs, x_in)  # (N, ns)

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(w_all.shape[1]):
        ax.plot(x, w_all[:, i])

    ax.set_title("Window weights")
    ax.grid(ls=":")


    # --- 在图下方添加子域和 overlap 可视化 ---
    y_bottom = -0.2  # y轴上的位置略低
    height = 0.08    # 色块高度

    for i, (xmin, xmax) in enumerate(zip(xmins, xmaxs)):
        xmin = float(xmin[0])
        xmax = float(xmax[0])
        width = xmax - xmin
        ov = width * float(overlap)

        # 主子域区域（不含 overlap）：灰色（排除左右 overlap）
        ax.add_patch(plt.Rectangle(
            (xmin + ov, y_bottom), width - 2 * ov, height,
            facecolor='lightgray', edgecolor='k', alpha=0.5
        ))

        # 左侧 overlap
        ax.add_patch(plt.Rectangle(
            (xmin, y_bottom), ov, height,
            facecolor='salmon', edgecolor='none', alpha=0.5
        ))

        # 右侧 overlap
        ax.add_patch(plt.Rectangle(
            (xmax - ov, y_bottom), ov, height,
            facecolor='salmon', edgecolor='none', alpha=0.5
        ))
    # 添加标注
    ax.text(x[0], y_bottom - 0.07, 'gray = subdomain, red = overlap', fontsize=8)

    # 调整 ylim 保证色块可见
    ax.set_ylim(y_bottom - 0.15, 1.05)

    if save_dir:
        path = os.path.join(save_dir, "window_weights.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path

# -------------------------------------------------
# 5. 训练统计保存
# -------------------------------------------------
def save_training_stats(loss_hist, test_steps, test_l1, save_dir):
    _ensure_dir(save_dir)
    jnp.savez(os.path.join(save_dir, "stats.npz"),
              loss_hist=loss_hist,
              test_steps=test_steps,
              test_l1=test_l1)
    print(f"[vis_1d] stats saved to {save_dir}/stats.npz")

# -------------------------------------------------
# 6. 将上面所有步骤打包成一个函数
# -------------------------------------------------
def visualize_1d(
    x_test:    jnp.ndarray,
    u_true:    jnp.ndarray,
    u_pred:    jnp.ndarray,
    loss_hist: jnp.ndarray,
    l1_steps:  jnp.ndarray,
    l1_hist:   jnp.ndarray,
    subdomains: dict = None,     # {"model": FBPINN, "list": subdomains_list}，若传 None 则跳过子域/窗口
    overlap:   float = None,     # 仅在绘制窗口函数时使用
    save_dir:  str = None,
):
    """
    一次性完成 1D 全部可视化：
      - Prediction vs Exact
      - Training Loss
      - Test L1 曲线
      - （可选）Subdomain contributions
      - （可选）Window weights

    Args:
    -----
      x_test      : shape (N,)
      u_true      : shape (N,)
      u_pred      : shape (N,)
      loss_hist   : shape (steps,)
      l1_steps    : shape (num_check,)
      l1_hist     : shape (num_check,)
      subdomains  : dict 或 None：
                    * 若是 dict，则包含两项：
                        "model" : FBPINN 对象
                        "list"  : subdomains_list
                    * 若是 None，则不画子域/窗口图
      overlap     : float, 子域 overlap ratio（仅绘制窗口时用）
      save_dir    : str，保存目录（可选）
    Returns:
    -------
      paths : dict，包含每张图对应的保存路径
    """
    _ensure_dir(save_dir)
    paths = {}

    # (1) Prediction vs Exact
    paths["pred_vs_exact"] = plot_prediction_vs_exact(
        x_test, u_true, u_pred, save_dir
    )

    # (2) Training loss
    paths["train_loss"] = plot_training_loss(
        loss_hist, save_dir
    )

    # (3) Test L1 curve
    paths["test_l1"] = plot_test_l1_curve(
        l1_steps, l1_hist, save_dir
    )

    # (4) 如果传入了 subdomains，就画子域贡献图与窗口函数图
    if subdomains is not None:
        fb_model      = subdomains["model"]
        subdomains_list = subdomains["list"]

        # (4.1) 检查是否有 subnets 属性，只有在 fb_model 是 FBPINN 时才画子域贡献图
        if hasattr(fb_model, 'subnets'):
            paths["subdomain_partials"] = plot_subdomain_partials(
                model    = fb_model,
                x_test   = x_test,
                u_true   = u_true,
                save_dir = save_dir,
            )

        # (4.2) 窗口函数权重
        paths["window_weights"] = plot_window_weights(
            x       = x_test,
            subdomains_list = subdomains_list,
            overlap = overlap,
            save_dir = save_dir,
        )

    # (5) 保存训练统计
    paths["stats"] = save_training_stats(
        loss_hist, l1_steps, l1_hist, save_dir
    )

    return paths


# -------------------------------------------------
# 7. 导出符号
# -------------------------------------------------
__all__ = [
    "plot_subdomain_partials",
    "plot_prediction_vs_exact",
    "plot_training_loss",
    "plot_test_l1_curve",
    "plot_window_weights",
    "save_training_stats",
    "visualize_1d",
]
