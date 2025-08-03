# utils/data_utils.py

import jax.numpy as jnp
import numpy as np
from scipy.stats.qmc import LatinHypercube, Halton, Sobol

def generate_subdomains(
    domain,
    overlap,
    centers_per_dim=None,      # list[array]  or None
    n_sub_per_dim=None,        # int | list[int] | None
    widths_per_dim=None,       # list[array|scalar] | None
):
    """
    生成 [(left, right), ...] 子域区间列表，支持：
      • 均匀中心  ——  n_sub_per_dim
      • 自定义中心 ——  centers_per_dim
      • 每维宽度  ——  widths_per_dim 可给标量或一维数组
    """
    lowers, uppers = map(jnp.asarray, domain)
    dim = lowers.size

    # ---------- 1. 生成中心 ----------
    if centers_per_dim is not None:
        axes = [jnp.asarray(c) for c in centers_per_dim]
    else:
        if isinstance(n_sub_per_dim, int):
            n_sub_per_dim = [n_sub_per_dim] * dim
        step = (uppers - lowers) / (jnp.array(n_sub_per_dim) - 1)
        axes = [
            jnp.linspace(lowers[i], uppers[i], int(n_sub_per_dim[i]))
            for i in range(dim)
        ]

    mesh = jnp.meshgrid(*axes, indexing="ij")                # 每个元素形状 (n0,n1,…)
    centers = jnp.stack([m.ravel() for m in mesh], axis=-1)  # (N, dim)

    grid_shape = mesh[0].shape                               # 基础形状
    N = centers.shape[0]

    # ---------- 2. half_width: broadcast 到整网格 ----------
    half_cols = []
    if widths_per_dim is not None:
        if len(widths_per_dim) != dim:
            raise ValueError("widths_per_dim 长度必须等于维度数")
        for d, w in enumerate(widths_per_dim):
            w_arr = jnp.asarray(w)

            # --- 标量 → 全域同值
            if w_arr.ndim == 0:
                w_full = jnp.full(grid_shape, w_arr)

            # --- 一维：长度必须与该维轴相同，reshape+broadcast
            elif w_arr.ndim == 1:
                if w_arr.shape[0] != axes[d].shape[0]:
                    raise ValueError(
                        f"widths_per_dim[{d}] 长度 {w_arr.shape[0]} "
                        f"与该维中心数 {axes[d].shape[0]} 不一致"
                    )
                shape = [1] * dim
                shape[d] = -1
                w_full = jnp.broadcast_to(w_arr.reshape(shape), grid_shape)

            # --- 已给 full grid：形状必须完全匹配
            else:
                if w_arr.shape != grid_shape:
                    raise ValueError(
                        f"widths_per_dim[{d}] 形状 {w_arr.shape} "
                        f"必须是标量 / 一维 / {grid_shape}"
                    )
                w_full = w_arr

            half_cols.append((w_full / 2.0).ravel())

    else:
        # === 未显式给 widths_per_dim ===
        if centers_per_dim is None:
            # 均匀网格：step + overlap
            step = (uppers - lowers) / (jnp.array(n_sub_per_dim) - 1)
            default_w = jnp.broadcast_to(step + overlap, (N, dim)) / 2.0
            half_cols = [default_w[:, d] for d in range(dim)]
        else:
            # 自定义中心：常数 overlap
            half_cols = [jnp.full(N, overlap / 2.0) for _ in range(dim)]

    half_width = jnp.stack(half_cols, axis=-1)                # (N, dim)

    # ---------- 3. 组装 ----------
    subdomains = [(c - hw, c + hw) for c, hw in zip(centers, half_width)]
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
    strategy: str = "grid",       # default grid
    seed: int = None,
    scramble: bool = False,
):
    """
    在 d 维超长方体内生成 n_pts^d 个 collocation 点，返回 ndarray (n_pts^d, d)。

    strategy 现支持:
      "grid"      – 规则网格 n_pts^d
      "uniform"   – 随机均匀分布
      "lhs"       – Latin Hypercube
      "halton"    – Halton 低差序列
      "sobol"     – Sobol 低差序列 (自动补 2^m 长度再截断)
      "hammersley"– 只做 2D 的 Hammersley
    """
    # -------- 1) 把 domain 规范成 [(lo,hi), ...] --------
    if isinstance(domain, tuple) and isinstance(domain[0], jnp.ndarray):
        lo_arr, hi_arr = domain
        lo_arr = jnp.asarray(lo_arr)
        hi_arr = jnp.asarray(hi_arr)
        assert lo_arr.shape == hi_arr.shape
        domain_list = [(float(lo_arr[i]), float(hi_arr[i])) for i in range(lo_arr.shape[0])]
    elif isinstance(domain[0], (float, int, np.floating)):
        domain_list = [(float(domain[0]), float(domain[1]))]     # 1-D
    else:
        domain_list = list(domain)                               # 已是 list[(lo,hi)]
    #print("domain_list",domain_list)
    d = len(domain_list)
    N = n_pts ** d

    rng = np.random.default_rng(seed)
    stg = strategy.lower()

    # -------- 2) 逐策略采样 --------
    if stg == "grid":
        axes = [np.linspace(lo, hi, n_pts) for lo, hi in domain_list]
        mesh = np.meshgrid(*axes, indexing="ij")
        pts  = np.column_stack([m.ravel() for m in mesh])        # (n_pts^d, d)

    elif stg == "uniform":
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
            raise NotImplementedError("Hammersley 仅支持 2D")
        pts_01 = np.zeros((N, 2))
        for i in range(N):
            pts_01[i, 0] = i / N
            pts_01[i, 1] = _van_der_corput(i, base=2)
        pts = _scale_sample(pts_01, domain_list)

    else:
        raise ValueError(f"Unknown strategy='{strategy}'. 请选择其中之一: "
                         "['grid','uniform','random','lhs','halton','sobol','hammersley'].")

    return pts


# =================== 测试 & 可视化 =================== #
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------
    # 1. generate_subdomains 回归 + 新功能测试
    # ------------------------------------------------------------
    print("===== generate_subdomains =====")

    # --- 1D：旧接口（n_sub_per_dim）
    domain_1d = (jnp.array([0.0]), jnp.array([1.0]))
    subs_1d = generate_subdomains(domain_1d,
                                  n_sub_per_dim=4,
                                  overlap=0.2)
    print("1D 均匀 4 子域:", len(subs_1d), subs_1d[:3], "...\n")

    # --- 1D：新接口（自定义中心）
    centers_1d = jnp.array([0.1, 0.4, 0.7, 0.95])
    subs_1d_custom = generate_subdomains(domain_1d,
                                         overlap=0.05,
                                         centers_per_dim=[centers_1d],
                                         widths_per_dim=[0.15])
    print("1D 自定义中心 4 子域:", len(subs_1d_custom), subs_1d_custom[:3], "...\n")

    # --- 2D：旧接口（均匀 5×5）
    domain_2d = (jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))
    subs_2d = generate_subdomains(domain_2d,
                                  n_sub_per_dim=[5, 5],
                                  overlap=0.1)
    print("2D 均匀 5×5 子域:", len(subs_2d), subs_2d[:3], "...\n")

    # --- 2D：新接口（6×8 峰 + 4 条贴边子域）
    x_centers = (jnp.arange(8) + 0.0) / 7.0             # 包含边界中心 0, 1
    y_centers = (jnp.arange(10) + 0.0) / 9.0
    widths_x  = jnp.array([1/6] + [2/6]*6 + [1/6])      # 贴边子域窄一些
    widths_y  = jnp.array([1/8] + [2/8]*8 + [1/8])
    subs_2d_custom = generate_subdomains(domain_2d,
                                         overlap=0.0,
                                         centers_per_dim=[x_centers, y_centers],
                                         widths_per_dim=[widths_x, widths_y])
    print("2D 自定义中心 + 宽度 子域:", len(subs_2d_custom),
          subs_2d_custom[:3], "...\n")

    # ------------------------------------------------------------
    # 2. generate_collocation 回归测试
    # ------------------------------------------------------------
    print("===== generate_collocation =====")

    # --- 1D, uniform
    pts1d = generate_collocation(domain_1d,
                                 n_pts=5,
                                 strategy="uniform")
    print("1D uniform 5 点:", pts1d.shape, pts1d)

    # --- 2D, grid
    pts2d = generate_collocation(domain_2d,
                                 n_pts=5,
                                 strategy="grid")
    print("2D grid 5×5 点:", pts2d.shape, "\n")

    # ------------------------------------------------------------
    # 3. 可视化不同 collocation strategy
    # ------------------------------------------------------------
    strategies = ["uniform", "grid", "lhs", "halton", "sobol", "hammersley"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    for ax, stg in zip(axes.flat, strategies):
        pts = generate_collocation(domain_2d,
                                   n_pts=8,
                                   strategy=stg,
                                   seed=42)
        ax.scatter(pts[:, 0], pts[:, 1], s=20, alpha=0.7,
                   edgecolors="k")
        ax.set_title(stg.capitalize())
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()
