import os
import sys
import yaml
import importlib
import jax
import jax.numpy as jnp
import optax
import equinox as eqx

# add path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import the model
from model.fbpinn_model import FBPINN
# import some data utils functions
from utils.data_utils import generate_collocation_points, generate_subdomain
# import some visualization functions
from utils.visualizer import (
    plot_prediction_vs_exact,
    plot_training_loss,
    plot_test_l1_curve,
    plot_window_weights,
    save_training_stats,
    plot_subdomain_partials
)
from tqdm import trange
from utils.data_utils  import generate_collocation_points


@eqx.filter_jit
def _step(model, opt_state, colloc_full, optimizer, residual_fn):
    def loss_fn(m):                                # full-batch residual
        return residual_fn(m, colloc_full)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    return eqx.apply_updates(model, updates), opt_state, loss

# Compute the l1 error using jax
@eqx.filter_jit
def compute_l1(model, x_test, u_test_exact):
    pred = jax.vmap(model)(x_test).squeeze()
    return jnp.mean(jnp.abs(pred - u_test_exact.squeeze()))


def train_fbpinn(
    *,
    model,
    subdomain_collocation_points, # list[n_sub] 
    steps,
    lr,
    pde_residual_loss,
    x_test          = None,
    u_exact         = None,
    save_dir        = None,
    checkpoint_every= 0,
):
    # 整包 collocation points（不再变化）
    colloc_full = subdomain_collocation_points

    # opt
    opt       = optax.adam(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    # training loop
    loss_hist, l1_hist, l1_steps = [], [], []
    pbar = trange(steps, desc="FBPINN", dynamic_ncols=True)

    for step in pbar:
        model, opt_state, loss = _step(model, opt_state,
                                       colloc_full, opt, pde_residual_loss)
        loss_val = float(loss); loss_hist.append(loss_val)

        # l1 error
        if x_test is not None and (step % 1 == 0 or step == steps-1):
            u_test_exact = u_exact(x_test) 
            l1 = float(compute_l1(model, x_test, u_test_exact))
            #l1 = float(jnp.mean(jnp.abs(jax.vmap(model)(x_test) - u_test_exact)))
            l1_hist.append(l1); l1_steps.append(step)
            pbar.set_postfix(loss=f"{loss_val:.2e}", l1=f"{l1:.2e}")
        else:
            pbar.set_postfix(loss=f"{loss_val:.2e}")

        # checkpoint
        if checkpoint_every and save_dir and (step+1) % checkpoint_every == 0:
            os.makedirs(save_dir, exist_ok=True)
            eqx.tree_serialise_leaves(
                os.path.join(save_dir, f"ckpt_{step+1}.eqx"), model
            )

    return model, jnp.array(loss_hist), (jnp.array(l1_steps), jnp.array(l1_hist))


# Test
if __name__ == "__main__":
    # Load Config
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "run.yaml")
    config_path = os.path.abspath(config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # PDE configuration
    pde_name = config["pde"]
    pde_module = importlib.import_module(f"physics.{pde_name}")
    u_exact = pde_module.u_exact
    ansatz = pde_module.ansatz
    pde_residual_loss = pde_module.pde_residual_loss
    domain = pde_module.DOMAIN

    # Training Configs
    steps = config["training"]["steps"]
    lr = config["training"]["lr"]
    n_sub = config["training"]["n_sub"]
    overlap = config["training"]["overlap"]
    n_points_per_subdomain = config["training"]["n_points_per_subdomain"]

    save_dir = config["save"]["output_dir"]
    ckpt_every = config["save"]["checkpoint_every"]

    activation_map = {
        "tanh": jax.nn.tanh,
        "relu": jax.nn.relu,
        "gelu": jax.nn.gelu,
        "sigmoid": jax.nn.sigmoid,
    }
    mlp_raw = config["mlp"]
    mlp_config = {
        **{k: mlp_raw[k] for k in ["in_size", "out_size", "width_size", "depth"]},
        "activation": activation_map[mlp_raw["activation"]],
    }

    # Generate subdomains
    subdomains_list = generate_subdomain(domain, n_sub, overlap)

    # Define Model
    key = jax.random.PRNGKey(42)
    model = FBPINN(
        key=key,
        num_subdomains=n_sub,
        ansatz=ansatz,
        subdomains=subdomains_list,
        mlp_config=mlp_config
    )

    # Generate Collocation Points
    subdomain_collocation_points, _ = generate_collocation_points(
        domain=domain,
        subdomains_list=subdomains_list,
        n_points_per_subdomain=n_points_per_subdomain,
        seed=0
    )

    # Test points (x_test)
    x_test = jnp.linspace(domain[0], domain[1], 300)

    # Train the model
    model, train_loss, (test_steps, test_l1) = train_fbpinn(
        model=model,
        subdomain_collocation_points=subdomain_collocation_points,
        steps=steps,
        lr=lr,
        x_test=x_test,
        save_dir=save_dir,
        checkpoint_every=ckpt_every,
        pde_residual_loss=pde_residual_loss,
        u_exact=u_exact
    )

    # Save Plots
    print("Done training. Saving plots.")
    u_pred = jax.vmap(model)(x_test)
    u_true = u_exact(x_test)

    plot_prediction_vs_exact(x_test, u_true, u_pred, save_dir)
    plot_training_loss(train_loss, save_dir)
    plot_test_l1_curve(test_steps, test_l1, save_dir)
    plot_window_weights(x_test, subdomains_list, n_sub, save_dir)
    plot_subdomain_partials(model, x_test, u_true, save_dir)
    save_training_stats(train_loss, test_steps, test_l1, save_dir)

    final_l1 = float(test_l1[-1]) if len(test_l1) > 0 else float('nan')
    print(f"Final L1 error = {final_l1:.4e}")
    print(f"Results saved to {save_dir}")
