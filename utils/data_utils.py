import jax
import jax.numpy as jnp
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
        step = total_len / n
        #centers = jnp.linspace(a + step / 2, b - step / 2, n)
        centers = jnp.linspace(a, b, n)
        print("centers",centers)
        grid_axes.append(centers)
        step_sizes.append(step)

    mesh = jnp.meshgrid(*grid_axes, indexing='ij')  # multi-D center coords
    center_points = jnp.stack([m.reshape(-1) for m in mesh], axis=-1)  # (n_sub_total, d)

    subdomains = []
    for center in center_points:
        width = jnp.array(step_sizes) + overlap/2
        left = center - width
        right = center + width
        #left = center - width / 2
        #right = center + width / 2
        subdomains.append((left, right))

    return subdomains


def generate_collocation_points(domain, subdomains_list, n_points_per_subdomain, seed=0):
    """Generate global collocation points and assign them to their respective subdomains."""
    n_sub = len(subdomains_list)
    n_total_collocation = n_sub * n_points_per_subdomain
    key = jax.random.PRNGKey(seed)

    # Sample uniformly within the global domain
    global_collocation_points = jax.random.uniform(
        key, (n_total_collocation,), minval=domain[0], maxval=domain[1]
    )

    # Assign points to subdomains
    subdomain_collocation_points = []
    for left, right in subdomains_list:
        mask = (global_collocation_points >= left) & (global_collocation_points <= right)
        points_in_subdomain = global_collocation_points[mask]
        subdomain_collocation_points.append(points_in_subdomain)
        #(f"Subdomain [{left:.2f}, {right:.2f}]: {len(points_in_subdomain)} points")

    return subdomain_collocation_points, global_collocation_points

if __name__ == "__main__":
    
    # 1D Test
    domain_1d = (jnp.array([0.0]), jnp.array([1.0]))
    overlap = 0.5
    n_sub = 2
    subdomains_1d = generate_subdomains(domain_1d, n_sub, overlap)
    print(subdomains_1d) #should be [(-0.25, 1.25), (0.75, 2.25)]
    
    n_points_per_subdomain = 500
    subdomain_collocation_points, global_collocation_points = generate_collocation_points(domain_1d, subdomains_1d, n_points_per_subdomain, seed=0)
    for i, subdomain in enumerate(subdomain_collocation_points):
        print(f"Subdomain {i} range: min = {min(subdomain)}, max = {max(subdomain)}")

    # 2D Test
    domain_2d = (jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))
    n_sub_2d = 2 # per dimension
    subdomains_2d = generate_subdomains(domain_2d, n_sub_2d, overlap)
    print(subdomains_2d)