import jax.numpy as jnp
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def triangle_wave(x, p=2):
    return 2 * jnp.abs(p * x - jnp.floor(p * x + 0.5))

#f (x) = exp[sin(0.3πx^2)]2

def toy_func(x):
    x = jnp.asarray(x)
    f = jnp.exp(jnp.sin(0.3 * jnp.pi * (x**2)))
    return f

def toy_func_2(x):
    x = jnp.asarray(x)
    f = jnp.sin((jnp.pi / 4.0) * x**2)
    return f

def triangle_wave_2(x,p=2):
    x = jnp.asarray(x)  
    f = jnp.zeros_like(x)  
    # x <= 0.2 f(x) = x / 0.2
    mask1 = x <= 0.2
    f[mask1] = x[mask1] / 0.2
    # x > 0.2，f(x) = 1.25 - 1.25 * x
    mask2 = x > 0.2
    f[mask2] = 1.25 - 1.25 * x[mask2]

    return f

class Normalizer:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def transform(self, x):
        """  x ——>[0,1] """
        return (x - self.min_val) / (self.max_val - self.min_val)

    def inverse_transform(self, x_scaled):
        """[0,1] to original """
        return x_scaled * (self.max_val - self.min_val) + self.min_val
    

def fit_local_polynomials_2nd(x, y, partitions): ### Second-order polynomials
    """
    Fit local second-order polynomials for each partition.
    """
    x_np = jnp.array(x)
    y_np = jnp.array(y)
    
    # Check the shape of partitions and ensure it is (N, C)
    partitions_np = jnp.array(partitions)
    print("Debug: partitions_np.shape", partitions_np.shape)

    # If partitions_np has more than 2 dimensions, squeeze it to 2D
    if len(partitions_np.shape) > 2:
        partitions_np = jnp.squeeze(partitions_np, axis=1)  # Remove the second singleton dimension
        print("Debug: after squeeze, partitions_np.shape", partitions_np.shape)

    # Ensure the shape is (N, C)
    N, C = partitions_np.shape  # N: number of data points, C: number of partitions
    print(f"Debug: N = {N}, C = {C}")

    coefficients = []
    for i in range(C):
        # Extract the i-th partition's weights for each data point (should be a 1D array of shape (N,))
        weights = partitions_np[:, i]
        
        # Create the design matrix [1, x, x^2]
        X = jnp.vstack([jnp.ones_like(x_np), x_np, x_np**2]).T  # shape [N, 3]
        
        # Create the weight matrix W as a diagonal matrix
        W = jnp.diag(weights + 1e-8)  # shape (N, N)

        # Weighted least squares: A = W @ X, b = W @ y
        A = W @ X  # shape (N, 3)
        b = W @ y_np  # shape (N,)

        # Solve using least squares (similar to np.linalg.lstsq)
        coeffs, _, _, _ = jnp.linalg.lstsq(A, b, rcond=None)
        
        # Append the coefficients for this partition
        coefficients.append(coeffs)

    return jnp.array(coefficients)

def visualize_final_approximation(
    model,
    params,
    x_train,
    y_train,
    normalizer=None,
    ground_truth_fn=None,
    show_partitions=True,
    save_path=None,
    title="Final Approximation",
    num_points=200
):
    """
    Visualize model prediction vs ground truth on original x-scale.

    Args:
        model: RBFPOUNet instance with .forward method
        params: Trained parameters dict with 'centers' and 'widths'
        x_train: Standardized training x (1D array in [0,1])
        y_train: Raw y values (not standardized)
        normalizer: Optional Normalizer, for inverse_transform
        ground_truth_fn: Optional function f(x) to plot as ground truth
        show_partitions: Bool, whether to show partition weights plot
        save_path: Path to save PNG file
        title: Title of the plot
        num_points: Number of test points to evaluate
    """
    # Make sure everything is JAX array
    x_train = jnp.array(x_train)
    y_train = jnp.array(y_train)

    # Predict partition weights on x_train
    partitions = model.forward(params["centers"], params["widths"], x_train)
    partitions_np = jnp.array(partitions)

    # Fit local polynomials
    coeffs = fit_local_polynomials_2nd(jnp.array(x_train), jnp.array(y_train), partitions_np)

    # Generate test x in normalized space
    x_test = jnp.linspace(0, 1, num_points)
    partitions_test = model.forward(params["centers"], params["widths"], x_test)
    partitions_test_np = jnp.array(partitions_test)

    # Evaluate prediction on test x
    y_test_pred = jnp.zeros_like(jnp.array(x_test))
    for i in range(partitions_test_np.shape[1]):
        c0, c1, c2 = coeffs[i]
        y_test_pred += partitions_test_np[:, i] * (c0 + c1 * x_test + c2 * x_test ** 2)

    # Rescale x to original scale if normalizer is provided
    if normalizer:
        x_train_plot = normalizer.inverse_transform(jnp.array(x_train))
        x_test_plot = normalizer.inverse_transform(jnp.array(x_test))
    else:
        x_train_plot = jnp.array(x_train)
        x_test_plot = jnp.array(x_test)

    # Plot prediction vs ground truth
    plt.figure(figsize=(7, 5))
    plt.plot(x_test_plot, y_test_pred, label="Model Approximation", color="red")

    # Plot ground truth if available
    if ground_truth_fn:
        x_gt = x_test_plot
        y_gt = ground_truth_fn(x_gt)
        plt.plot(x_gt, y_gt, "--", label="Ground Truth", color="black")
        # Compute and show MSE
        mse = jnp.mean((y_gt - y_test_pred) ** 2)
        plt.title(f"{title}\nMSE = {mse:.4e}")
    else:
        plt.title(title)

    # Plot training points
    plt.scatter(x_train_plot, jnp.array(y_train), marker='x', color="black", label="Training Points")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    # Optional: Show partition weights
    if show_partitions:
        plt.figure(figsize=(7, 4))
        for i in range(partitions_test_np.shape[1]):
            plt.plot(x_test_plot, partitions_test_np[:, i])
        plt.title("Partition-of-Unity Weights")
        plt.xlabel("x")
        plt.ylabel("Weight")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        if save_path:
            part_path = save_path.replace(".png", "_pou.png")
            plt.savefig(part_path, dpi=300)
            print(f"Saved partition plot to {part_path}")
        else:
            plt.show()
