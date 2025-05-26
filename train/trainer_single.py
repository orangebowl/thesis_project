import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import os,sys
import matplotlib.pyplot as plt
from tqdm import trange  # For progress bar

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Function to compute loss for a mini-batch of points
def loss_fn(model, x_collocation, pde_residual):
    return pde_residual(model, x_collocation)

@eqx.filter_jit
def train_step_single(model, opt_state, x_collocation, optimizer, pde_residual):
    """Perform a single training step for the PINN."""
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model, x_collocation, pde_residual)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val

@eqx.filter_jit
def compute_l1(model, x_test, u_test_exact):
    """Compute L1 error on the test set."""
    pred = jax.vmap(model)(x_test).squeeze()
    return jnp.mean(jnp.abs(pred - u_test_exact.squeeze()))

def train_single(
    model,
    colloc,
    lr,
    steps,
    pde_residual,
    batch_size,
    x_test=None,
    u_exact=None,
    save_dir=None,
    checkpoint_every=0,
):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Shuffle collocation points to create random mini-batches
    num_batches = len(colloc) // batch_size
    if num_batches == 0:
        raise ValueError("Batch size is larger than the number of collocation points.")

    # Tracking loss and L1 errors
    loss_list = []
    l1_list = []
    loss_hist = []
    l1_hist = []
    l1_steps = []  # Steps where L1 error is computed
    pbar = trange(steps, desc="PINN", dynamic_ncols=True)

    for i in pbar:
        # Shuffle indices for mini-batches
        shuffled_indices = jax.random.permutation(jax.random.PRNGKey(i), len(colloc))
        for batch_idx in range(num_batches):
            # Get mini-batch of collocation points
            batch_indices = shuffled_indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            x_batch = colloc[batch_indices]

            # Perform one training step
            model, opt_state, loss_val = train_step_single(model, opt_state, x_batch, optimizer, pde_residual)

            loss_list.append(loss_val)

            # Calculate test L1 error at intervals
            if x_test is not None and u_exact is not None and (batch_idx % 10 == 0 or i == steps - 1):
                u_test_exact = u_exact(x_test)
                l1_error = float(compute_l1(model, x_test, u_test_exact))
                l1_list.append(l1_error)
                l1_steps.append(i)  # Save the current step for L1 calculation

                pbar.set_postfix(loss=f"{loss_val:.2e}", l1=f"{l1_error:.2e}")
            else:
                # Ensure we don't update l1_list when no L1 error is computed
                continue

        loss_hist.append(loss_list[-1])  # Record the last loss value
        #l1_hist.append(l1_list[-1])  # Record the last L1 error value
        if l1_list:                        # ← 新增保护
            l1_hist.append(l1_list[-1])    # 只有在真的算过 L1 时才记录

        # Save checkpoint
        if checkpoint_every and save_dir and (i + 1) % checkpoint_every == 0:
            os.makedirs(save_dir, exist_ok=True)
            eqx.tree_serialise_leaves(
                os.path.join(save_dir, f"ckpt_{i+1}.eqx"), model
            )

    # Ensure `t_steps` and `t_l1` are always aligned
    t_steps = jnp.array(l1_steps)
    t_l1 = jnp.array(l1_list)

    # Check if `t_steps` and `t_l1` lengths match
    if len(t_steps) != len(t_l1):
        raise ValueError(f"Mismatch in lengths of t_steps and t_l1. t_steps: {len(t_steps)}, t_l1: {len(t_l1)}")

    return model, jnp.array(loss_hist), (t_steps, t_l1)





# Test 
if __name__ == "__main__":
    import os,sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from model.pinn_model import PINN
    from physics.pde_cosine import u_exact, pde_residual_loss, ansatz
    import jax.random as jr

    save_dir = "outputs/single_pinn_cosine_15"
    os.makedirs(save_dir, exist_ok=True)

    key = jr.PRNGKey(0)
    model = PINN(key, ansatz)

    x_collocation = jnp.linspace(-2 * jnp.pi, 2 * jnp.pi, 200)
    x_test = jnp.linspace(-2 * jnp.pi, 2 * jnp.pi, 200)

    # 训练
    model, train_loss, test_l1 = train_single(
        model,
        x_collocation,
        lr=1e-3,
        steps=5000,
        pde_residual=pde_residual_loss,
        x_test=x_test,
        u_exact=u_exact,
    )

    eqx.tree_serialise_leaves(os.path.join(save_dir, "final_model.eqx"), model)

    jnp.save(os.path.join(save_dir, "train_loss.npy"), train_loss)
    jnp.save(os.path.join(save_dir, "test_l1.npy"), test_l1)

    u_pred = jax.vmap(model)(x_test)
    u_true = u_exact(x_test)

    plt.figure()
    plt.plot(x_test, u_true, "--", label="Exact")
    plt.plot(x_test, u_pred, label="PINN Pred")
    plt.title("PINN Prediction vs Exact")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "prediction_vs_exact.png"), dpi=300)

    # visulize training loss & L1 test loss
    plt.figure()
    plt.plot(train_loss)
    plt.yscale("log")
    plt.title("Training Loss Curve")
    plt.xlabel("Steps")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=300)

    plt.figure()
    plt.plot(test_l1)
    plt.yscale("log")  
    plt.title("Test L1 Curve")
    plt.xlabel("Steps")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "test_l1_curve.png"), dpi=300)
    print("", save_dir)
