import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def loss_fn(model, x_collocation, pde_residual):
    # 这里已经是一整个批次，不需要再 vmap
    return pde_residual(model, x_collocation)


@eqx.filter_jit
def train_step_single(model, opt_state, x_collocation, optimizer, pde_residual):
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model, x_collocation, pde_residual)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val

def train_single(model, x_collocation, lr, steps, pde_residual, x_test=None, u_exact=None):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    loss_list = []
    l1_list = []

    for i in range(steps):
        model, opt_state, loss_val = train_step_single(model, opt_state, x_collocation, optimizer, pde_residual)
        loss_list.append(loss_val)

        if x_test is not None and u_exact is not None:
            u_pred = jax.vmap(model)(x_test)
            u_true = u_exact(x_test)
            l1_error = jnp.mean(jnp.abs(u_pred - u_true))
            l1_list.append(l1_error)
            if i % 1000 == 0 or i == steps - 1:
                print(f"[SinglePINN] Step={i}, Loss={loss_val:.3e}, Test L1={l1_error:.4e}")
        else:
            l1_list.append(jnp.nan)

    return model, jnp.array(loss_list), jnp.array(l1_list)


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
