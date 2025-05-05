import argparse, importlib, yaml
from pathlib import Path
import jax, jax.numpy as jnp
from utils.visualizer import (
    plot_prediction_vs_exact, plot_training_loss, plot_test_l1_curve,
    plot_window_weights, plot_subdomain_partials, save_training_stats
)

# ────────────────────── Load Configuration ──────────────────────
def load_config(config_path):
    """Load configuration from YAML file."""
    return yaml.safe_load(Path(config_path).read_text())

# ────────────────────── Build Model ──────────────────────
def build_model(CFG, prob, key, mlp_cfg):
    """Return model and training module based on configuration."""
    if CFG["model_type"].lower() == "fbpinn":
        from model.fbpinn_model import FBPINN
        from utils.data_utils import generate_subdomain, generate_collocation_points
        # Build FBPINN model
        n_sub = CFG["training"]["n_sub"]
        overlap = CFG["training"]["overlap"]
        subs = generate_subdomain(prob.domain, n_sub, overlap)
        model = FBPINN(key, n_sub, prob.ansatz, subs, mlp_cfg)

        n_pts = CFG["training"]["n_points_per_subdomain"]
        colloc, _ = generate_collocation_points(
            domain=prob.domain, subdomains_list=subs,
            n_points_per_subdomain=n_pts, seed=0
        )
        trainer_mod = "train.trainer_fbpinn"
    else:
        from model.pinn_model import PINN
        model = PINN(key, prob.ansatz, width=mlp_cfg["width_size"], depth=mlp_cfg["depth"])
        total = CFG["training"]["n_sub"] * CFG["training"]["n_points_per_subdomain"]
        colloc = jnp.linspace(*prob.domain, total)
        trainer_mod = "train.trainer_single"

    return subs, model, colloc, trainer_mod

# ────────────────────── Train Model ──────────────────────
def train_model(model, colloc, trainer_mod, RESIDUAL, CFG, prob):
    """Train the model and return the results."""
    TrainMod = importlib.import_module(trainer_mod)
    train_fn = getattr(TrainMod, "train_fbpinn" if trainer_mod.endswith("fbpinn")
                                else "train_single")

    steps, lr = CFG["training"]["steps"], CFG["training"]["lr"]
    out_dir = Path(CFG["save"]["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if trainer_mod.endswith("fbpinn"):
        # FBPINN
        model, loss_hist, (t_steps, t_l1) = train_fn(
            model=model,
            subdomain_collocation_points=colloc,
            steps=steps,
            lr=lr,
            pde_residual_loss=RESIDUAL,
            x_test=jnp.linspace(*prob.domain, 300),
            u_exact=prob.exact,
            save_dir=str(out_dir),
            checkpoint_every=CFG["save"].get("checkpoint_every", 0)
        )
    else:
        # PINN
        model, loss_hist, t_l1 = train_fn(
            model, colloc, lr, steps,
            RESIDUAL, jnp.linspace(*prob.domain, 300), prob.exact,
            batch_size=CFG["training"]["batch_size"]
        )
        t_steps = jnp.arange(len(t_l1))

    return model, loss_hist, t_l1, t_steps, out_dir

# ────────────────────── Visualize Results ──────────────────────
def visualize_results(loss_hist, t_steps, t_l1, subs, model, prob, out_dir):
    """Visualize and save results."""
    x = jnp.linspace(*prob.domain, 500)
    u_pred, u_true = jax.vmap(model)(x), prob.exact(x)

    plot_prediction_vs_exact(x, u_true, u_pred, out_dir)
    plot_training_loss(loss_hist, out_dir)
    plot_test_l1_curve(t_steps, t_l1, out_dir)

    if len(subs) > 0:
        plot_window_weights(x, subs, len(subs), out_dir)
        plot_subdomain_partials(model, x, u_true, out_dir)

    save_training_stats(loss_hist, t_steps, t_l1, out_dir)
    print(f"✓ Done. Results saved to {out_dir}")

# ────────────────────── Main Function ──────────────────────
def main():
    # Parse command line arguments and load config
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config/run.yaml", help="YAML config file path")
    args = ap.parse_args()
    CFG = load_config(args.cfg)

    # Load physics problem and model config
    mod_path, cls_name = CFG["pde"].rsplit(".", 1)
    ProbCls = getattr(importlib.import_module(f"physics.{mod_path}"), cls_name)
    prob = ProbCls()
    DOMAIN = prob.domain
    RESIDUAL = prob.residual

    # MLP configuration
    act_map = {"tanh": jax.nn.tanh, "relu": jax.nn.relu, "gelu": jax.nn.gelu, "sigmoid": jax.nn.sigmoid}
    mlp_cfg = dict(CFG["mlp"])
    mlp_cfg["activation"] = act_map[mlp_cfg["activation"]]

    # Set random key
    key = jax.random.PRNGKey(0)

    # Build model and training module
    subs, model, colloc, trainer_mod = build_model(CFG, prob, key, mlp_cfg)

    # Train the model
    model, loss_hist, t_l1, t_steps, out_dir = train_model(model, colloc, trainer_mod, RESIDUAL, CFG, prob)

    # Visualize and save results
    visualize_results(loss_hist, t_steps, t_l1, subs, model, prob, out_dir)

if __name__ == "__main__":
    main()
