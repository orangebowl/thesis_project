import jax
import jax.numpy as jnp
"""
functions that used in window.py to generate window functions
"""

def generate_subdomain(domain, n_sub, overlap):
    """generate uniform subdomain according to the domain, 
    number of subdomains and the length of overlap"""
    total_len = domain[1] - domain[0]
    step_size = total_len / n_sub
    width = step_size + overlap

    centers = jnp.linspace(domain[0] + step_size / 2,
                           domain[1] - step_size / 2,
                           n_sub)
    
    subdomains_list = []
    for i in range(n_sub):
        left = float(centers[i] - width / 2)
        right = float(centers[i] + width / 2)
        subdomains_list.append((left, right)) # tuple list
    return subdomains_list 

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
    domain = (0,2)
    n_sub = 2
    overlap = 0.5
    subdomains_list = generate_subdomain(domain, n_sub, overlap)
    print(subdomains_list) #should be [(-0.25, 1.25), (0.75, 2.25)]
    
    n_points_per_subdomain = 500
    subdomain_collocation_points, global_collocation_points = generate_collocation_points(domain, subdomains_list, n_points_per_subdomain, seed=0)
    for i, subdomain in enumerate(subdomain_collocation_points):
        print(f"Subdomain {i} range: min = {min(subdomain)}, max = {max(subdomain)}")
