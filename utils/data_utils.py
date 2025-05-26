import jax.numpy as jnp
import numpy as np
from scipy.stats.qmc import LatinHypercube, Halton, Sobol
import matplotlib.pyplot as plt
"""
functions that used in window.py to generate window functions
"""

def generate_subdomains(domain, n_sub_per_dim, overlap):
    """
    Generate uniformly spaced subdomains (with overlap) in 1D or multi-D.

    Args:
        domain: tuple of (lower_bounds: jnp.array, upper_bounds: jnp.array), shape (d,)
        n_sub_per_dim: int or list of ints for each dim
        overlap: float (relative size, e.g., 0.2)

    Returns:
        subdomains: list of (left: jnp.array, right: jnp.array)
    """
    if isinstance(n_sub_per_dim, int):
        n_sub_per_dim = [n_sub_per_dim] * len(domain[0])  # scalar → list

    dim = len(domain[0])
    grid_axes = []
    step_sizes = []
    for i in range(dim):
        a, b = domain[0][i], domain[1][i]
        n = n_sub_per_dim[i]
        total_len = b - a
        step = total_len / (n-1)
        #centers = jnp.linspace(a + step / 2, b - step / 2, n)
        centers = jnp.linspace(a, b, n)
        print("centers",centers)
        grid_axes.append(centers)
        step_sizes.append(step)

    mesh = jnp.meshgrid(*grid_axes, indexing='ij')  # multi-D center coords
    center_points = jnp.stack([m.reshape(-1) for m in mesh], axis=-1)  # (n_sub_total, d)

    subdomains = []
    for center in center_points:
        width = jnp.array(step_sizes)/2 + overlap/2 # half
        left = center - width
        right = center + width
        #left = center - width / 2
        #right = center + width / 2
        subdomains.append((left, right))

    return subdomains


### Different sampling strategy
# =================== 辅助函数 =================== #
def scale_sample(samples: np.ndarray, domain):
    """
    将在 [0,1]^2 上的采样点映射回给定的 domain = [(x_lo, y_lo), (x_hi, y_hi)].
    """
    (x_lo, y_lo), (x_hi, y_hi) = domain
    return samples * np.array([x_hi - x_lo, y_hi - y_lo]) + np.array([x_lo, y_lo])

def _van_der_corput(index, base=2):
    """
    范德科蒙 (Van der Corput) 展开，将整数 index 映射为 [0,1] 之间的小数。
    用于 Hammersley 序列等低差异采样。
    """
    result = 0.0
    f = 1.0
    i = index
    while i > 0:
        f /= base
        result += f * (i % base)
        i //= base
    return result

# =================== 主函数：采样入口 =================== #
def generate_collocation(domain, n_pts, strategy="random", seed=None, scramble=False):
    """
    根据 strategy 生成 n_pts^2 个二维采样点，返回 shape=(n_pts^2, 2).
    可选的 strategy:
        - "uniform"    : 均匀网格采样
        - "random"     : 随机
        - "lhs"        : Latin Hypercube
        - "halton"     : Halton  (支持 scramble)
        - "sobol"      : Sobol   (支持 scramble)
        - "hammersley" : Hammersley
    其余参数:
        - seed       : 随机种子(对 random / lhs / halton / sobol 有效)
        - scramble   : 是否扰动(对 halton / sobol 有效)
    """

    (x_lo, y_lo), (x_hi, y_hi) = domain
    N = n_pts ** 2  # 采样总数

    # 根据 strategy 分支
    if strategy.lower() == "uniform":
        # 均匀网格
        xs = np.linspace(x_lo, x_hi, n_pts)
        ys = np.linspace(y_lo, y_hi, n_pts)
        XX, YY = np.meshgrid(xs, ys, indexing='ij')
        points = np.column_stack([XX.ravel(), YY.ravel()])

    elif strategy.lower() == "random":
        # 随机采样
        rng = np.random.default_rng(seed)
        samples_01 = rng.random((N, 2))
        points = scale_sample(samples_01, domain)

    elif strategy.lower() == "lhs":
        # Latin Hypercube
        sampler = LatinHypercube(d=2, seed=seed)
        samples_01 = sampler.random(N)
        points = scale_sample(samples_01, domain)

    elif strategy.lower() == "halton":
        # Halton (可 scramble)
        sampler = Halton(d=2, scramble=scramble, seed=seed)
        samples_01 = sampler.random(N)
        points = scale_sample(samples_01, domain)

    elif strategy.lower() == "sobol":
        # Sobol (可 scramble)
        sampler = Sobol(d=2, scramble=scramble, seed=seed)
        samples_01 = sampler.random(N)
        points = scale_sample(samples_01, domain)

    elif strategy.lower() == "hammersley":
        # Hammersley
        points_01 = np.zeros((N, 2))
        for i in range(N):
            points_01[i, 0] = i / N
            points_01[i, 1] = _van_der_corput(i, base=2)
        points = scale_sample(points_01, domain)

    else:
        raise ValueError(f"Unknown strategy={strategy}. Choose from "
                         f"['uniform','random','lhs','halton','sobol','hammersley'].")

    return points

# =================== 测试 & 可视化 =================== #
if __name__ == "__main__":
    domain = ((0, 0), (1, 1))
    n_pts  = 4  # 采样 4^2=16 个点，方便可视化

    # 不同采样方式
    strategies = [
        "uniform",
        "random",
        "lhs",
        "halton",
        "sobol",
        "hammersley"
    ]

    # 生成并打印
    for stg in strategies:
        pts = generate_collocation(domain, n_pts, strategy=stg, seed=42, scramble=False)
        print(f"\nStrategy: {stg}")
        print(pts)

    # 画图对比
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)

    for ax, stg in zip(axes.flat, strategies):
        pts = generate_collocation(domain, n_pts, strategy=stg, seed=42, scramble=False)
        ax.scatter(pts[:, 0], pts[:, 1], s=30, alpha=0.7, edgecolors='k')
        ax.set_title(stg.capitalize())
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.show()