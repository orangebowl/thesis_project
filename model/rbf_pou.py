import jax
import jax.numpy as jnp

class RBFPOUNet:
    """
    A simple Radial Basis Function (RBF) Partition of Unity (POU) network.
    This network generates partitions of unity using Gaussian RBFs.
    """

    def __init__(self, input_dim=1, num_centers=5, width_val=0.4, key=jax.random.PRNGKey(0)):
        """
        Constructor to initialize the RBF POU network.

        Args:
            input_dim (int): Dimensionality of the input features. Default is 1.
            num_centers (int): Number of RBF centers. Default is 5.
            width_val (float): Initial width (or scale) for each RBF. Default is 0.4.
            key (jax.random.PRNGKey): A PRNGKey for random number generation.
        """
        self.num_centers = num_centers

        # Randomly initialize the centers in [0, 1], shape = (num_centers, input_dim)
        self.centers = jax.random.uniform(key, (num_centers, input_dim), minval=0.0, maxval=1.0)

        # Initialize all widths to the same value, shape = (num_centers,)
        #self.widths = jnp.ones(num_centers) * width_val
        self.widths = jax.random.uniform(key, (num_centers,), minval=0.05, maxval=0.5)

    def init_params(self):
        """
        Returns a parameter dictionary with centers and widths 
        that were randomly initialized in the constructor.
        
        Returns:
            dict: 
                {
                    "centers": (num_centers, input_dim),
                    "widths": (num_centers,)
                }
        """
        return {
            "centers": self.centers,
            "widths": self.widths
        }

    def init_params_fixed(self):
        """
        Returns a parameter dictionary where the centers are equally spaced in [0, 1] 
        (reshaped to (num_centers, 1)) and the widths are fixed to 0.01 for all centers.

        Returns:
            dict:
                {
                    "centers": (num_centers, 1),
                    "widths": (num_centers,)
                }
        """
        # Create equally spaced centers in [0, 1]
        centers = jnp.linspace(0.0, 1.0, self.num_centers).reshape(-1, 1)
        # Set all widths to 0.01
        widths = self.widths
        
        return {
            "centers": centers,
            "widths": widths
        }
        
    @staticmethod
    def forward(centers, widths, x):
        """
        Compute the partition of unity from given centers, widths, and inputs x.

        The output partitions ensure that, for each x_i:
            sum_{j=1..num_centers} partition_{ij} ~= 1

        Args:
            centers (jnp.ndarray): Shape (num_centers, input_dim).
            widths (jnp.ndarray): Shape (num_centers,).
            x (jnp.ndarray): Input array, shape (N,) if input_dim=1 
                             or (N, input_dim) if input_dim>1.

        Returns:
            jnp.ndarray: The POU array of shape (N, num_centers), 
                         where each row sums to approximately 1.
        """
        # If x is 1D, reshape to (N, 1) to handle broadcast correctly
        x = x[:, None]  # shape (N, 1)
        
        # Broadcast x and centers for element-wise operations
        x_broadcasted = x[:, None, :]             # shape (N, 1, 1)
        centers_broadcasted = centers[None, :, :] # shape (1, num_centers, 1)
        
        # Compute squared distances (N, num_centers)
        diff = x_broadcasted - centers_broadcasted  
        dist_sq = jnp.sum(diff ** 2, axis=2)     
        
        # Broadcast widths to match (N, num_centers)
        widths_broadcasted = widths[None, :]       # shape (1, num_centers)
        
        # Exponential radial basis function
        rbf_vals = jnp.exp(-dist_sq / (widths_broadcasted ** 2))
        
        # Normalize so that each row sums to 1
        rbf_sum = jnp.sum(rbf_vals, axis=1, keepdims=True) + 1e-9
        partitions = rbf_vals / rbf_sum
        return partitions


if __name__ == "__main__":
    # Example usage:

    # 1) Create an RBFPOUNet with default width_val=0.4
    rbf_net_default = RBFPOUNet(input_dim=1, num_centers=5)
    params_def = rbf_net_default.init_params()

    # Prepare an input array x in [0, 1]
    x = jnp.linspace(0, 1, 20)
    partitions_def = rbf_net_default.forward(params_def["centers"], params_def["widths"], x)

    print("Default initialization:")
    print("Centers:\n", params_def["centers"])
    print("Widths:\n", params_def["widths"])
    print("Sum of partitions per input:\n", jnp.sum(partitions_def, axis=1))

    # 2) Create an RBFPOUNet with a custom width_val=0.01
    rbf_net_small_width = RBFPOUNet(input_dim=1, num_centers=5, width_val=0.01)
    params_small = rbf_net_small_width.init_params()
    partitions_small = rbf_net_small_width.forward(params_small["centers"], params_small["widths"], x)

    print("\nCustom width=0.01 initialization:")
    print("Centers:\n", params_small["centers"])
    print("Widths:\n", params_small["widths"])
    print("Sum of partitions per input:\n", jnp.sum(partitions_small, axis=1))

    # 3) Use the fixed initialization method (centers equally spaced, widths=0.01)
    rbf_net_fixed = RBFPOUNet(input_dim=1, num_centers=5, key=jax.random.PRNGKey(42))
    params_fixed = rbf_net_fixed.init_params_fixed()
    partitions_fixed = rbf_net_fixed.forward(params_fixed["centers"], params_fixed["widths"], x)

    print("\nFixed initialization (linspace centers, width=0.01):")
    print("Centers:\n", params_fixed["centers"])
    print("Widths:\n", params_fixed["widths"])
    print("Sum of partitions per input:\n", jnp.sum(partitions_fixed, axis=1))
