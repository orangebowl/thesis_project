# viz_utils.py
"""
可视化辅助：仅包含绘图函数。
其余训练 / 网络代码仍在原脚本。
"""

from __future__ import annotations
import math, pathlib
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Sequence
import os
from pathlib import Path
from datetime import datetime

# --------- 复制你脚本里用到的三个 helper ------------------
from pou_all import (_design_matrix, fit_local_polynomials,
                            _predict_from_coeffs, _toy_func)  # 保证路径正确
# ---------------------------------------------------------

SAVE_DIR = pathlib.Path("visualizations")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import math
from pathlib import Path

# Ensure SAVE_DIR is defined and exists
SAVE_DIR = Path("output")  # Example, make sure this exists or create it dynamically

def viz_partitions(model, params, title="partitions", grid=200):
    """
    Visualize learned partitions of the model. 
    Supports 1D and 2D partitions.
    
    Args:
        model: The model object with a forward method.
        params: Parameters for the model.
        title (str): The title of the plot.
        grid (int): The resolution of the grid for visualization.
        
    Returns:
        str: The file name where the plot is saved.
    """
    d = model.input_dim  # Assuming the model has an attribute input_dim
    fname = SAVE_DIR / f"{title.replace(' ', '_')}.png"

    # Ensure the SAVE_DIR exists
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    if d == 1:
        # 1D case: Plot the weights for each point on the grid
        xs = jnp.linspace(0, 1, grid)[:, None]
        part = model.forward(params, xs)
        
        plt.figure(figsize=(6, 3))
        plt.plot(xs.squeeze(), part)
        plt.xlabel("x")
        plt.ylabel("weight")
        plt.title(title)
        plt.tight_layout()

    else:  # d == 2
        # 2D case: Visualize each partition as an image
        xs = jnp.linspace(0, 1, grid)
        xx, yy = jnp.meshgrid(xs, xs)
        pts = jnp.stack([xx.ravel(), yy.ravel()], -1)
        part = model.forward(params, pts)
        
        C = part.shape[1]
        cols = int(math.ceil(math.sqrt(C)))
        rows = int(math.ceil(C / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        
        # Ensure axes is always iterable
        if isinstance(axes, plt.Axes):
            axes = [axes]  # If it's a single Axes object, make it iterable
        else:
            axes = axes.ravel()  # Flatten if it's a 2D array of axes

        for i in range(C):
            axes[i].imshow(part[:, i].reshape(grid, grid),
                           origin="lower", extent=[0, 1, 0, 1],
                           cmap="viridis", vmin=0, vmax=1)
            axes[i].set_title(f"part {i}")
            axes[i].set_xticks([]); axes[i].set_yticks([])

        # Hide any unused axes
        for j in range(C, len(axes)):
            axes[j].axis("off")

        fig.suptitle(title)
        plt.tight_layout()

    # Save the plot
    plt.savefig(fname, dpi=300)
    plt.close()
    
    print(f"Saved partition visualization to {fname}")
    
    return fname  # Return the file path for further use



def viz_final(model, params, x_train, y_train, grid=200):
    part_tr = model.forward(params, x_train)
    coeffs  = fit_local_polynomials(x_train, y_train, part_tr, lam=0.0)
    d       = model.input_dim
    fname   = SAVE_DIR / f"final_approx_{d}d.png"

    if d == 1:
        xs = jnp.linspace(0, 1, grid)[:, None]
        part   = model.forward(params, xs)
        y_pred = _predict_from_coeffs(xs, coeffs, part)
        plt.figure(figsize=(6, 3))
        plt.plot(xs.squeeze(), _toy_func(xs), "--", label="truth")
        plt.plot(xs.squeeze(), y_pred, label="prediction")
        plt.legend(); plt.tight_layout()
    else:
        xs = jnp.linspace(0, 1, grid)
        xx, yy = jnp.meshgrid(xs, xs)
        pts = jnp.stack([xx.ravel(), yy.ravel()], -1)
        part   = model.forward(params, pts)
        y_pred = _predict_from_coeffs(pts, coeffs, part).reshape(grid, grid)
        y_true = _toy_func(pts).reshape(grid, grid)
        err    = jnp.abs(y_true - y_pred)
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        for a, z, t in zip(ax, [y_pred, y_true, err],
                           ["pred", "truth", "abs err"]):
            im = a.imshow(z, cmap="viridis", origin="lower",
                          extent=[0, 1, 0, 1])
            a.set_title(t)
            plt.colorbar(im, ax=a, shrink=.8)
        plt.tight_layout()

    plt.savefig(fname, dpi=300); plt.close()
    return fname



def visualize_pou_windows(
    net,
    pou_params: Any,
    x_test: np.ndarray,
    domain: Sequence[Tuple[float, float]],
    test_n: int,
    N_SUB: int,
    *,
    cmap: str = "inferno",
    verify: bool = True,
    title: str = "Visualization of Learned PoU Windows",
    save_path: str | Path | None = None,
    save_dpi: int = 300,
    show: bool = True,
):
    """Plot the learnt Partition-of-Unity window functions.

    This is a lightly enhanced version (adds *save_path*, *show*) of the
    original Jupyter snippet so it can be called from non-interactive scripts.
    """

    # 1. Compute window weights
    pou_weights = net.forward(pou_params, x_test)  # (num_points, N_SUB)

    # 2. Figure layout
    if N_SUB == 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()  # Ensure axes is always iterable
    elif N_SUB == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    else:
        cols = min(N_SUB, 4)
        rows = int(np.ceil(N_SUB / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        axes = axes.ravel()  # Ensure axes is always iterable

    fig.suptitle(title, fontsize=16)
    extent = [domain[0][0], domain[1][0], domain[0][1], domain[1][1]]

    # 3. Plot each window
    for i in range(N_SUB):
        ax = axes[i]
        window_grid = pou_weights[:, i].reshape(test_n, test_n)
        im = ax.imshow(
            window_grid, extent=extent, origin="lower",
            cmap=cmap, vmin=0, vmax=1,
        )
        ax.set_title(f"PoU Window {i+1}", fontsize=14)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes (when N_SUB < total number of axes)
    for j in range(N_SUB, len(axes)):
        axes[j].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 4. Optional save
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
        print(f"[{datetime.now():%H:%M:%S}] PoU windows saved → {save_path}")

    # 5. PoU verification
    if verify:
        sums = pou_weights.sum(axis=1)
        print("\nVerifying Partition-of-Unity property:")
        print(f"  First 5 sums : {sums[:5]}")
        print(f"  Mean         : {sums.mean():.6f}")
        print(f"  Std dev      : {sums.std():.6f}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_colloc_points(colloc, domain, stage_id, save_dir="results/rad_stages"):
    """
    绘制并保存在每个RAD阶段采样到的配置点。(此函数无需修改)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(6,6))
    plt.scatter(colloc[:, 0], colloc[:, 1], s=12, alpha=0.6, edgecolors='k', linewidth=0.5)
    plt.xlim(domain[0][0], domain[1][0])
    plt.ylim(domain[0][1], domain[1][1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"RAD Collocation Points (After Stage {stage_id})")
    plt.savefig(os.path.join(save_dir, f"rad_colloc_stage_{stage_id}.png"))
    plt.show()
    
# ──────────────────────────────────────────────────────────────────────────────
# Final field comparison: exact vs prediction vs absolute error
# ──────────────────────────────────────────────────────────────────────────────

def visualize_final_results(
    u_true: np.ndarray,
    u_pred: np.ndarray,
    domain: Sequence[Tuple[float, float]],
    test_n: int,
    *,
    title: str = "Comparison of Solutions and Error",
    cmap_solution: str = "viridis",
    cmap_error: str = "Reds",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot exact solution, FBPINN prediction, and absolute error.

    Parameters
    ----------
    u_true, u_pred
        1‑D arrays of length ``test_n ** 2`` (flattened meshes).
    domain
        ``((x_min, x_max), (y_min, y_max))``.
    test_n
        Number of points per spatial dimension used to build the meshgrid.
    title, cmap_solution, cmap_error
        Customise appearance.
    save_path
        If provided, save the figure to this file (parent dirs auto‑created).
    show
        Whether to call :pyfunc:`matplotlib.pyplot.show`. If ``False`` the
        figure is closed (for non‑blocking batch scripts).
    """
    # ── Reshape field data ───────────────────────────────────────────────────
    U_true_grid = u_true.reshape(test_n, test_n)
    U_pred_grid = u_pred.reshape(test_n, test_n)
    Error_grid  = np.abs(U_true_grid - U_pred_grid)

    extent = [domain[0][0], domain[1][0], domain[0][1], domain[1][1]]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    fig.suptitle(title, fontsize=16)

    # Exact solution
    im0 = axes[0].imshow(U_true_grid, extent=extent, origin="lower",
                         cmap=cmap_solution)
    axes[0].set_title("Exact Solution", fontsize=14)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Prediction
    im1 = axes[1].imshow(U_pred_grid, extent=extent, origin="lower",
                         cmap=cmap_solution)
    axes[1].set_title("FBPINN‑PoU Prediction", fontsize=14)
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Absolute error
    im2 = axes[2].imshow(Error_grid, extent=extent, origin="lower",
                         cmap=cmap_error)
    axes[2].set_title("Absolute Error", fontsize=14)
    axes[2].set_xlabel("x"); axes[2].set_ylabel("y")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # ── Save / show / close ──────────────────────────────────────────────────
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[{datetime.now():%H:%M:%S}] Solution comparison saved → {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Training & test‑loss history
# ──────────────────────────────────────────────────────────────────────────────

def visualize_training_history(
    loss_hist: np.ndarray,
    l1_steps: np.ndarray,
    l1_hist: np.ndarray,
    *,
    title: str = "Training & L1 Test Loss History",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot training PDE residual and L1 test loss on dual y‑axes (log‑scale)."""

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Training loss (left y‑axis)
    p1, = ax1.plot(loss_hist, label="Training Loss (PDE Residual)",
                   color="deepskyblue")
    ax1.set_xlabel("Training Steps", fontsize=14)
    ax1.set_ylabel("Training Loss (Log Scale)", fontsize=14,
                   color="deepskyblue")
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor="deepskyblue")
    ax1.grid(True, which="both", ls="--", color="lightgrey")

    # L1 test loss (right y‑axis) — share x
    ax2 = ax1.twinx()
    p2, = ax2.plot(l1_steps, l1_hist, linestyle="--", marker="o", markersize=4,
                   label="L1 Test Loss", color="orangered")
    ax2.set_ylabel("L1 Test Loss (Log Scale)", fontsize=14, color="orangered")
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor="orangered")

    # Legend & title
    fig.legend(handles=[p1, p2], loc="upper right", bbox_to_anchor=(0.9, 0.9))
    fig.suptitle(title, fontsize=16)

    # ── Save / show / close ──────────────────────────────────────────────────
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[{datetime.now():%H:%M:%S}] Loss history saved → {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Utility to dump stats for later analysis / plotting
# ──────────────────────────────────────────────────────────────────────────────

def save_stats(
    path: str | Path,
    *,
    loss_hist: np.ndarray,
    l1_steps: np.ndarray,
    l1_hist: np.ndarray,
    u_true_final: np.ndarray,
    u_pred_pou: np.ndarray,
) -> Path:
    """Save arrays to a *.npz* bundle for later inspection."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        loss_hist=loss_hist,
        l1_steps=l1_steps,
        l1_hist=l1_hist,
        u_true_final=u_true_final,
        u_pred_pou=u_pred_pou,
    )
    print(f"[{datetime.now():%H:%M:%S}] Stats saved → {path}")
    return path