# utils/data_utils.py

import jax.numpy as jnp
import numpy as np
from scipy.stats.qmc import LatinHypercube, Halton, Sobol

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


def _scale_sample(samples_01, domain):
    """
    把 [0,1]^d 的样本样本样本，缩放到给定 domain。
    domain: list of (lo, hi)；其中 lo, hi 均为数值 (float) 或 np.float。
    """
    lows = np.array([lo for lo, _ in domain])  # shape (d,)
    highs = np.array([hi for _, hi in domain])
    return lows + samples_01 * (highs - lows)   # shape (N, d)


def _van_der_corput(n, base=2):
    """1-D Van-der-Corput 序列，用于 Hammersley 2D 采样。"""
    vdc, denom = 0.0, 1.0
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder / denom
    return vdc


def generate_collocation(
    domain,
    n_pts: int,
    strategy: str = "random",
    seed: int = None,
    scramble: bool = False,
):
    """
    在 d-维超长方体 domain 内生成 n_pts^d 个 collocation 点。返回 ndarray 形状 (n_pts^d, d)。

    Args:
        domain: 
            - 1D 情况: (jnp.array([xmin]), jnp.array([xmax])) 
            - 2D 及更高维: (jnp.array([xmin, ymin, ...]), jnp.array([xmax, ymax, ...]))
            也可以传列表形式：[(xmin,xmax), (ymin,ymax), ...]，等价于上面数组版。
        n_pts: 每个维度上的采样点数 (总点数 = n_pts**d)
        strategy: "uniform"、"random"、"lhs"、"halton"、"sobol"、"hammersley"
        seed: 随机种子 (仅对 "random"/"lhs"/"halton"/"sobol" 有效)
        scramble: 是否对 Halton 或 Sobol 做 scramble

    Returns:
        pts: NumPy 数组，shape = (n_pts**d, d)，每行是一个 d 维采样坐标。
    """
    # 1) 先把 domain 规范成列表 [(lo,hi), (lo,hi), ...] 形式
    if isinstance(domain, tuple) and isinstance(domain[0], jnp.ndarray):
        # numpy/jax 数组版，直接把单项归为列表
        lo_arr, hi_arr = domain
        lo_arr = jnp.asarray(lo_arr)
        hi_arr = jnp.asarray(hi_arr)
        assert lo_arr.shape == hi_arr.shape, "domain 两个数组必须同形状 (d,)"
        d = lo_arr.shape[0]
        domain_list = [(float(lo_arr[i]), float(hi_arr[i])) for i in range(d)]
    elif isinstance(domain[0], (float, int, np.floating)):
        # domain=(xmin,xmax) 1D
        domain_list = [(float(domain[0]), float(domain[1]))]
    else:
        # domain 已经是 list-of-tuples 形式
        domain_list = list(domain)
        d = len(domain_list)

    d = len(domain_list)
    N = n_pts ** d

    rng = np.random.default_rng(seed)
    stg = strategy.lower()

    if stg == "uniform":
        # 每维 n_pts 均匀节点 → meshgrid → 展平
        axes = [np.linspace(lo, hi, n_pts) for lo, hi in domain_list]
        mesh = np.meshgrid(*axes, indexing="ij")
        pts = np.column_stack([m.ravel() for m in mesh])

    elif stg == "random":
        samples_01 = rng.random((N, d))
        pts = _scale_sample(samples_01, domain_list)

    elif stg == "lhs":
        sampler = LatinHypercube(d=d, seed=seed)
        samples_01 = sampler.random(N)
        pts = _scale_sample(samples_01, domain_list)

    elif stg == "halton":
        sampler = Halton(d=d, scramble=scramble, seed=seed)
        samples_01 = sampler.random(N)
        pts = _scale_sample(samples_01, domain_list)

    elif stg == "sobol":
        sampler = Sobol(d=d, scramble=scramble, seed=seed)
        m = int(np.ceil(np.log2(N)))
        samples_01 = sampler.random(2 ** m)[:N]
        pts = _scale_sample(samples_01, domain_list)

    elif stg == "hammersley":
        if d != 2:
            raise NotImplementedError("Hammersley 仅支持 2D。")
        pts_01 = np.zeros((N, 2))
        for i in range(N):
            pts_01[i, 0] = i / N
            pts_01[i, 1] = _van_der_corput(i, base=2)
        pts = _scale_sample(pts_01, domain_list)

    else:
        raise ValueError(
            f"Unknown strategy='{strategy}'. "
            "请从 ['uniform','random','lhs','halton','sobol','hammersley'] 中选择。"
        )

    return pts


# =================== 测试 & 可视化 =================== #
if __name__ == "__main__":
    # ---------- 测试 generate_subdomains ----------
    domain_1d = (jnp.array([0.0]), jnp.array([1.0]))  # 1D
    subs_1d = generate_subdomains(domain_1d, n_sub_per_dim=4, overlap=0.2)
    print("1D 子域数量:", len(subs_1d), "示例：", subs_1d[:2])

    domain_2d = (jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))  # 2D
    subs_2d = generate_subdomains(domain_2d, n_sub_per_dim=[3, 2], overlap=0.1)
    print("2D 子域数量:", len(subs_2d), "示例：", subs_2d[:2])

    # ---------- 测试 generate_collocation ----------
    # 1D 情况
    pts1d = generate_collocation((jnp.array([0.0]), jnp.array([1.0])), n_pts=5, strategy="uniform")
    print("1D 均匀 5 点, 形状:", pts1d.shape, pts1d[:5])

    # 2D 情况：数组形式
    pts2d_arr = generate_collocation((jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0])), n_pts=5, strategy="uniform")
    print("2D 数组域 均匀 5×5, 形状:", pts2d_arr.shape, pts2d_arr[:5])

    # 2D 情况：列表形式
    pts2d_list = generate_collocation([(0.0, 0.0), (1.0, 1.0)], n_pts=5, strategy="random", seed=42)
    print("2D 列表域 随机 5×5, 形状:", pts2d_list.shape, pts2d_list[:5])

    # 可视化对比（任选一种）
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    strategies = ["uniform", "random", "lhs", "halton", "sobol", "hammersley"]
    for ax, stg in zip(axes.flat, strategies):
        pts = generate_collocation((jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0])), n_pts=8, strategy=stg, seed=42)
        ax.scatter(pts[:, 0], pts[:, 1], s=20, alpha=0.7, edgecolors='k')
        ax.set_title(stg.capitalize())
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()
