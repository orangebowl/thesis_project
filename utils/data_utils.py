import jax.numpy as jnp
import numpy as np
from scipy.stats.qmc import LatinHypercube, Halton, Sobol

def generate_subdomains(
    domain,
    overlap,
    centers_per_dim=None,      # list[array]  or None
    n_sub_per_dim=None,        # int | list[int] | None
    widths_per_dim=None,       # list[array|scalar] | None
    verbose: bool = False,
):
    lowers, uppers = map(jnp.asarray, domain)
    dim = lowers.size

    # ---------- 1. Centers ----------
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

    mesh = jnp.meshgrid(*axes, indexing="ij")
    centers = jnp.stack([m.ravel() for m in mesh], axis=-1)  # (N, dim)
    grid_shape = mesh[0].shape
    N = centers.shape[0]

    # ---------- 2. half_width ----------
    half_cols = []
    if widths_per_dim is not None:
        if len(widths_per_dim) != dim:
            raise ValueError("The length of widths_per_dim must equal the number of dimensions")
        for d, w in enumerate(widths_per_dim):
            w_arr = jnp.asarray(w)
            if w_arr.ndim == 0:
                w_full = jnp.full(grid_shape, w_arr)
            elif w_arr.ndim == 1:
                if w_arr.shape[0] != axes[d].shape[0]:
                    raise ValueError(
                        f"The length of widths_per_dim[{d}] ({w_arr.shape[0]}) "
                        f"does not match the number of centers in that dimension ({axes[d].shape[0]})"
                    )
                shape = [1] * dim
                shape[d] = -1
                w_full = jnp.broadcast_to(w_arr.reshape(shape), grid_shape)
            else:
                if w_arr.shape != grid_shape:
                    raise ValueError(
                        f"The shape of widths_per_dim[{d}] ({w_arr.shape}) must be scalar, 1D, or {grid_shape}"
                    )
                w_full = w_arr
            half_cols.append((w_full / 2.0).ravel())
    else:
        if centers_per_dim is None:
            # Uniform grid: step + overlap (overlap as absolute widening)
            step = (uppers - lowers) / (jnp.array(n_sub_per_dim) - 1)
            default_w = jnp.broadcast_to(step + overlap, (N, dim)) / 2.0
            half_cols = [default_w[:, d] for d in range(dim)]
        else:
            # Custom centers: treat overlap as the absolute value of the "total width"
            half_cols = [jnp.full(N, overlap / 2.0) for _ in range(dim)]

    half_width = jnp.stack(half_cols, axis=-1)  # (N, dim)

    # ---------- 3. Assembly ----------
    subdomains = [(c - hw, c + hw) for c, hw in zip(centers, half_width)]

    # ---------- 4. Optional Printing ----------
    if verbose:
        for i, (left, right) in enumerate(subdomains):
            l = np.asarray(left, dtype=float).ravel()
            r = np.asarray(right, dtype=float).ravel()
            size = r - l
            l_str = "(" + ", ".join(f"{v:.6g}" for v in l) + ")"
            r_str = "(" + ", ".join(f"{v:.6g}" for v in r) + ")"
            sz_str = "(" + "×".join(f"{v:.6g}" for v in size) + ")"
            print(f"[{i:03d}] left={l_str}, right={r_str}, size={sz_str}")

    return subdomains

def _scale_sample(samples_01, domain):
    """
    Scales samples from [0,1]^d to the given domain.
    domain: list of (lo, hi), where lo, hi are numeric (float) or np.float.
    """
    lows = np.array([lo for lo, _ in domain])   # shape (d,)
    highs = np.array([hi for _, hi in domain])
    return lows + samples_01 * (highs - lows)   # shape (N, d)


def _van_der_corput(n, base=2):
    """1-D Van-der-Corput sequence, used for Hammersley 2D sampling."""
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
    Generates n_pts^d collocation points in a d-dimensional hyperrectangle.

    Returns:
        ndarray (n_pts^d, d)

    Strategies supported:
      "grid"      – Regular grid with n_pts^d points
      "uniform"   – Random uniform distribution
      "lhs"       – Latin Hypercube Sampling
      "halton"    – Halton low-discrepancy sequence
      "sobol"     – Sobol low-discrepancy sequence (padded to 2^m then truncated)
      "hammersley"– Hammersley sequence (2D only)
    """
    # -------- 1) Normalize domain to [(lo,hi), ...] --------
    if isinstance(domain, tuple) and isinstance(domain[0], jnp.ndarray):
        lo_arr, hi_arr = map(jnp.asarray, domain)
        assert lo_arr.shape == hi_arr.shape
        domain_list = [(float(lo_arr[i]), float(hi_arr[i])) for i in range(lo_arr.shape[0])]
    elif isinstance(domain[0], (float, int, np.floating)):
        domain_list = [(float(domain[0]), float(domain[1]))]  # 1-D
    else:
        domain_list = list(domain)                            # Already list[(lo,hi)]
    
    d = len(domain_list)
    N = n_pts ** d

    rng = np.random.default_rng(seed)
    stg = strategy.lower()

    # -------- 2) Sample based on strategy --------
    if stg == "grid":
        axes = [np.linspace(lo, hi, n_pts) for lo, hi in domain_list]
        mesh = np.meshgrid(*axes, indexing="ij")
        pts = np.column_stack([m.ravel() for m in mesh])  # (n_pts^d, d)

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
            raise NotImplementedError("Hammersley sampling is implemented for 2D only")
        pts_01 = np.zeros((N, 2))
        for i in range(N):
            pts_01[i, 0] = i / N
            pts_01[i, 1] = _van_der_corput(i, base=2)
        pts = _scale_sample(pts_01, domain_list)

    else:
        raise ValueError(f"Unknown strategy='{strategy}'. Please choose from: "
                         "['grid', 'uniform', 'lhs', 'halton', 'sobol', 'hammersley'].")

    return pts

def generate_subdomains_zeros(
    domain,
    *,
    n_zeros_per_dim: int = 11,
    overlap_abs: float = 0.06,
    verbose: bool = False,
):
    """
    Generates non-uniform subdomains ensuring a constant overlap of overlap_abs.
    
    The method calculates the left/right half-widths for each subdomain in each
    dimension separately and clips them against the domain boundaries. This ensures
    that overlaps near the edges are not reduced.

    Returns:
        A list of subdomains: [(left, right), ...].
    """
    lo_vec, hi_vec = map(jnp.asarray, domain)
    D = int(lo_vec.size)
    if n_zeros_per_dim < 2:
        raise ValueError("n_zeros_per_dim must be at least 2.")

    m = int(n_zeros_per_dim)
    add = float(overlap_abs)
    add2 = 0.5 * add  # Half-width to add on each side

    centers_axes = []
    halfL_axes = []  # Leftward half-widths for each dimension
    halfR_axes = []  # Rightward half-widths for each dimension

    for d in range(D):
        lo, hi = float(lo_vec[d]), float(hi_vec[d])

        # Zeros and base segments
        unit_zeros = jnp.sqrt(jnp.arange(m) / (m - 1.0))  # in [0,1]
        zeros = lo + (hi - lo) * unit_zeros
        l, r = zeros[:-1], zeros[1:]
        centers_1d = 0.5 * (l + r)
        base_half = 0.5 * (r - l)  # Base half-width

        # Target half-width = base half-width + overlap_abs/2
        desired = base_half + add2

        # In this simplified version, clipping is removed. The half-widths are uniform.
        halfL_1d = desired
        halfR_1d = desired

        centers_axes.append(centers_1d)  # (m-1,)
        halfL_axes.append(halfL_1d)      # (m-1,)
        halfR_axes.append(halfR_1d)      # (m-1,)

    # Create grid
    mesh_centers = jnp.meshgrid(*centers_axes, indexing="ij")
    grid_shape = mesh_centers[0].shape

    # Broadcast left and right half-widths to the full grid
    halfL_grids, halfR_grids = [], []
    for d in range(D):
        shape = [1] * D
        shape[d] = -1
        halfL_grids.append(jnp.broadcast_to(halfL_axes[d].reshape(shape), grid_shape))
        halfR_grids.append(jnp.broadcast_to(halfR_axes[d].reshape(shape), grid_shape))

    # Flatten
    centers_mat = jnp.stack([mc.ravel() for mc in mesh_centers], axis=-1)  # (N,D)
    halfL_mat = jnp.stack([g.ravel() for g in halfL_grids], axis=-1)    # (N,D)
    halfR_mat = jnp.stack([g.ravel() for g in halfR_grids], axis=-1)    # (N,D)

    left = centers_mat - halfL_mat
    right = centers_mat + halfR_mat

    subdomains = [(left[i], right[i]) for i in range(left.shape[0])]

    # Optional printing during generation
    if verbose:
        for i, (lft, rgt) in enumerate(subdomains):
            l = np.asarray(lft, dtype=float).ravel()
            r = np.asarray(rgt, dtype=float).ravel()
            size = r - l
            l_str = "(" + ", ".join(f"{v:.6g}" for v in l) + ")"
            r_str = "(" + ", ".join(f"{v:.6g}" for v in r) + ")"
            sz_str = "(" + "×".join(f"{v:.6g}" for v in size) + ")"
            print(f"[{i:03d}] left={l_str}, right={r_str}, size={sz_str}")

    return subdomains