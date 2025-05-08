import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

def my_window_func(subdomains_tuple, n_sub, x, tol=1e-8):
    """
    Implements the FBPINN window function using shifted sigmoid centers:
    
        w_i(x) = Π_j sigmoid((x_j - μ_min^j)/σ_j) * sigmoid((μ_max^j - x_j)/σ_j)

    Args:
        subdomains_tuple: List of (left: jnp.array, right: jnp.array), shape (n_sub, 2, d)
        n_sub: Number of subdomains
        x: shape (N, d), input points
        tol: Sigmoid tail threshold (controls sharpness)

    Returns:
        w_norm: shape (N, n_sub), normalized window weights
    """
    x = jnp.atleast_2d(x)  # (N, d)
    N, d = x.shape

    # Subdomain bounds
    a = jnp.stack([sd[0] for sd in subdomains_tuple])  # (n_sub, d)
    b = jnp.stack([sd[1] for sd in subdomains_tuple])  # (n_sub, d)
    width = b - a                                      # (n_sub, d)

    # Sigmoid smoothing parameter: σ = width / (2 * t)
    t = jnp.log((1 - tol) / tol)
    sigma = width / (2 * t)  # shape (n_sub, d)

    # Shifted sigmoid centers (same as FBPINN reference)
    mu_min = a + width / 4
    mu_max = b - width / 4

    # Expand x: (N, 1, d)
    x_exp = x[:, None, :]  # (N, 1, d)

    # Compute product of left & right sigmoids across dimensions
    left_sig = jax.nn.sigmoid((x_exp - mu_min) / sigma)    # (N, n_sub, d)
    right_sig = jax.nn.sigmoid((mu_max - x_exp) / sigma)   # (N, n_sub, d)
    prod = left_sig * right_sig                            # (N, n_sub, d)

    w_raw = jnp.prod(prod, axis=-1)                        # (N, n_sub)
    w_norm = w_raw / (jnp.sum(w_raw, axis=1, keepdims=True) + 1e-10)
    return w_norm





# 放在 utils/test_window.py 中或你的 main 脚本底部
if __name__ == "__main__":
    import os, sys
    import matplotlib.pyplot as plt
    import jax.numpy as jnp
    import jax
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from utils.window_function import my_window_func
    from data_utils import generate_subdomains

    # params
    overlap = 0.2
    tol = 1e-8
    n_sub_1d = 2
    n_sub_2d = 3

    # 2D test
    domain_2d = (jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))
    subdomains_2d = generate_subdomains(domain_2d, n_sub_2d, overlap)

    print("2D Subdomains:", subdomains_2d)

    x_test_2d = jnp.array([[0.25, 0.25], [0.75, 0.75], [0.5, 0.5]])
    w_2d = my_window_func(subdomains_2d, len(subdomains_2d), x_test_2d, tol=tol)
    print("2D Window weights:\n", w_2d)
    print("2D Row sums (should ≈ 1.0):", jnp.sum(w_2d, axis=1))

    def plot_window_weights_2d(subdomains, target_sub=0, n=100):
        x = jnp.linspace(0, 1, n)
        y = jnp.linspace(0, 1, n)
        xx, yy = jnp.meshgrid(x, y)
        pts = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)
        w = my_window_func(subdomains, len(subdomains), pts, tol=tol)
        w_target = w[:, target_sub].reshape(n, n)

        plt.figure(figsize=(5, 5))
        plt.imshow(w_target, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
        plt.title(f"2D Window Weights: Subdomain {target_sub} (tol={tol})")
        plt.colorbar(label="Weight")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(False)
        plt.show()

    print("Testing 2D window weight heatmaps:")
    for i in range(len(subdomains_2d)):
        plot_window_weights_2d(subdomains_2d, target_sub=i)

    #1D test
    domain_1d = (jnp.array([0.0]), jnp.array([1.0]))
    subdomains_1d = generate_subdomains(domain_1d, n_sub_1d, overlap)

    print("1D Subdomains:", subdomains_1d)

    def plot_window_weights_1d(subdomains, n_points=300):
        x = jnp.linspace(0, 1, n_points).reshape(-1, 1)
        w = my_window_func(subdomains, len(subdomains), x, tol=tol)

        plt.figure(figsize=(8, 4))
        for i in range(w.shape[1]):
            plt.plot(x.squeeze(), w[:, i], label=f"Subdomain {i}")
        plt.title(f"1D Window Weights per Subdomain (tol={tol})")
        plt.xlabel("x")
        plt.ylabel("Weight")
        plt.legend()
        plt.grid(True)
        plt.show()

    print("Testing 1D window weight curves:")
    plot_window_weights_1d(subdomains_1d)

