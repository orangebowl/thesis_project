import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def my_window_func(subdomains_tuple, n_sub, x, tol=1e-8):
    """
    Compute the window function for multiple subdomains (batch, n_sub),
    used as fixed weights that do not participate in backpropagation.

    Args:
        subdomains_tuple: Python tuple/list, each element: (left, righ)
        n_sub: number of subdomains
        x: shape (batch,), current batch of points
        tol: float, controls the tail decay of the sigmoid function

    Returns:
        w_norm: shape (batch, n_sub), window weights
    """
    x = jnp.atleast_1d(x)  # (batch,)

    # JAX array a, b (shape=(n_sub,))
    left_list = [sd[0] for sd in subdomains_tuple]
    right_list = [sd[1] for sd in subdomains_tuple]
    a = jnp.array(left_list)  # (n_sub,)
    b = jnp.array(right_list) # (n_sub,)

    # compute width of each subdomain
    width = b - a   # (n_sub,)

    # sigmoid parameters
    t = jnp.log((1 - tol) / tol)
    mu_min = a + width / 2
    mu_max = b - width / 2
    sd = width / (2 * t)

    # x => (batch, 1)
    x_2d = x[:, None]  # (batch, 1)

    left_sig = jax.nn.sigmoid((x_2d - mu_min) / sd)   # shape (batch, n_sub)
    right_sig = jax.nn.sigmoid((mu_max - x_2d) / sd)  # shape (batch, n_sub)

    w_raw = left_sig * right_sig   # (batch, n_sub)
    sum_w = jnp.sum(w_raw, axis=1, keepdims=True) + 1e-10
    w_norm = w_raw / sum_w         # (batch, n_sub)
    return w_norm


if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from utils.data_utils import generate_subdomain
    
    x_test = jnp.linspace(-2 * jnp.pi, 2 * jnp.pi, 300)
    domain = (-2 * jnp.pi, 2 * jnp.pi)
    n_sub = 5
    overlap = 4
    tol = 1e-8
    subdomains_tuple = generate_subdomain(domain=domain,n_sub=n_sub,overlap=overlap)
    # compute window weights
    window_weights = my_window_func(subdomains_tuple, n_sub, x_test, tol=tol)
    # window_weights shape: (300, n_sub)

    # plot
    plt.figure(figsize=(10, 6))
    for i in range(window_weights.shape[1]):
        plt.plot(x_test, window_weights[:, i], label=f"Subdomain {i}")

    plt.xlabel('x')
    plt.ylabel('Window Weight')
    plt.legend()
    plt.title('Window Weights for Each Subdomain')
    plt.grid(True)
    plt.show()
