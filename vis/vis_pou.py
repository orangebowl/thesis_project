# viz_utils.py
"""
可视化辅助：仅包含绘图函数。
其余训练 / 网络代码仍在原脚本。
"""

from __future__ import annotations
import math, pathlib
import jax.numpy as jnp
import matplotlib.pyplot as plt

# --------- 复制你脚本里用到的三个 helper ------------------
from pou_all import (_design_matrix, fit_local_polynomials,
                            _predict_from_coeffs, _toy_func)  # 保证路径正确
# ---------------------------------------------------------

SAVE_DIR = pathlib.Path("visualizations")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

def viz_partitions(model, params, title="partitions", grid=200):
    d = model.input_dim
    fname = SAVE_DIR / f"{title.replace(' ', '_')}.png"

    if d == 1:
        xs = jnp.linspace(0, 1, grid)[:, None]
        part = model.forward(params, xs)
        plt.figure(figsize=(6, 3))
        plt.plot(xs.squeeze(), part)
        plt.xlabel("x"); plt.ylabel("weight"); plt.title(title)
        plt.tight_layout()
    else:                               # d == 2
        xs = jnp.linspace(0, 1, grid)
        xx, yy = jnp.meshgrid(xs, xs)
        pts = jnp.stack([xx.ravel(), yy.ravel()], -1)
        part = model.forward(params, pts)
        C = part.shape[1]
        cols = int(math.ceil(math.sqrt(C)))
        rows = int(math.ceil(C / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        axes = axes.ravel()
        for i in range(C):
            axes[i].imshow(part[:, i].reshape(grid, grid),
                           origin="lower", extent=[0, 1, 0, 1],
                           cmap="viridis", vmin=0, vmax=1)
            axes[i].set_title(f"part {i}")
            axes[i].set_xticks([]); axes[i].set_yticks([])
        for j in range(C, len(axes)): axes[j].axis("off")
        fig.suptitle(title); plt.tight_layout()

    plt.savefig(fname, dpi=300); plt.close()
    return fname                      # 如需在 notebook 中显示可用

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
