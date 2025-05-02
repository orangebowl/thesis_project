import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.rbf_pou import RBFPOUNet
from utils.pou_utils import toy_func, Normalizer, fit_local_polynomials_2nd

def visualize_pou_weights(forward_fn, params, x_train, epoch, phase):
    x_jnp = jnp.array(x_train)
    with jax.disable_jit():
        partitions = forward_fn(params["centers"], params["widths"], x_jnp)
    partitions_np = jnp.array(partitions)
    plt.figure(figsize=(4, 4))
    for i in range(partitions_np.shape[1]):
        #plt.plot(x_train, partitions_np[:, i], label=f"Partition {i+1}")
        plt.plot(x_train, partitions_np[:, i])
    plt.xlabel("x")
    plt.ylabel("Partition Weight")
    plt.title(f"Phase {phase}, Epoch {epoch}")
    plt.legend()
    plt.ylim([0, 1])
    plt.show()

# Two-phase LSGD training
def train_two_phase_lsgd_rbf(model, x_train, y_train, num_epochs_phase1=1000, num_epochs_phase2=3000,
                             lambda_reg=0.1, rho=0.99, n_stag=200, lr_phase1=0.1, lr_phase2=0.05, fixed_pou = None):
    
    if fixed_pou:
        params = model.init_params_fixed() # Fixed POU initial guess
    else:
        params = model.init_params()

    @jax.jit
    def loss_fn(params, x, y, reg_lambda):
        partitions = model.forward(params["centers"], params["widths"], x)
        coeffs = fit_local_polynomials_2nd(x, y, partitions)
        y_pred = jnp.zeros_like(y)
        for i in range(partitions.shape[1]):
            c0, c1, c2 = coeffs[i]
            y_pred += partitions[:, i] * (c0 + c1 * x + c2 * x**2)
        mse_loss = jnp.mean((y_pred - y) ** 2 + 1e-8)
        reg_loss = reg_lambda * (jnp.mean(params["widths"]**2))  # reg
        return mse_loss + reg_loss

    @jax.jit
    def grad_fn(params, reg_lambda):
        return jax.grad(loss_fn)(params, x_train, y_train, reg_lambda)

    # Track the best loss
    best_loss = float('inf')
    stagnation_counter = 0

    for phase in range(2):
        if phase == 0:
            reg_lambda = lambda_reg
            learning_rate = lr_phase1
            num_epochs = num_epochs_phase1
        else:
            reg_lambda = 0.0
            learning_rate = lr_phase2
            num_epochs = num_epochs_phase2

        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)

        for epoch in range(num_epochs):
            grads = grad_fn(params, reg_lambda)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            current_loss = loss_fn(params, x_train, y_train, reg_lambda)

            if current_loss < best_loss:
                best_loss = current_loss
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > n_stag:
                reg_lambda *= rho
                stagnation_counter = 0

            # Visualization
            if epoch % (num_epochs // 10) == 0:
                visualize_pou_weights(model.forward, params, x_train, epoch, phase+1)

            if epoch % 10 == 0:
                print(f"Phase {phase+1}, Epoch {epoch}, Loss: {current_loss:.6f}, λ: {reg_lambda:.6f}")

    return params

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    model = RBFPOUNet(input_dim=1, num_centers=3, key=key)

    x_train_raw = jnp.linspace(0, 8, 100)  # original scale

    normalizer = Normalizer(min_val=x_train_raw.min(), max_val=x_train_raw.max())
    x_train = normalizer.transform(x_train_raw)  # normalized to [0,1]
    y_train = toy_func(x_train)

    final_params = train_two_phase_lsgd_rbf(
        model,
        x_train,
        y_train,
        num_epochs_phase1=1000,
        num_epochs_phase2=500,
        lambda_reg=0.2,
        rho=0.99,
        n_stag=50,
        lr_phase1=0.001,
        lr_phase2=0.0005
    )