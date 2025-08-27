import os
import datetime
import jax
import jax.numpy as jnp
from typing import Dict, Any

from utils.data_utils import (
    generate_subdomains,
    generate_subdomains_zeros,
    generate_collocation,
)
from physics.problems import CosineODE, FirstOrderFreq1010
from model.fbpinn_model import FBPINN
from model.pinn_model import PINN
from train.trainer_fbpinn import train_fbpinn
from train.trainer_single import train_pinn
from vis.vis_1d import visualize_1d
from vis.vis_2d import visualize_2d, save_training_stats, plot_subdomains


def main(cfg: Dict[str, Any]) -> None:
    # Output directory
    save_dir = os.path.join("ckpts", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved in {os.path.abspath(save_dir)}")

    # Problem selection
    problem = cfg["pde_module"]()
    u_exact = problem.exact
    ansatz = problem.ansatz
    domain = problem.domain
    xdim = getattr(domain[0], "size", 1)
    print(f"Problem Dimension: {xdim}D")

    # Model construction
    key = jax.random.PRNGKey(cfg["seed"])
    mlp_conf = dict(
        in_size=xdim,
        out_size=1,
        width_size=cfg["width_size"],
        depth=cfg["depth"],
        activation=jax.nn.tanh,
    )

    model_type = cfg["model_type"].lower()
    if model_type == "fbpinn":
        subdomain_strategy = cfg.get("subdomain_strategy", "uniform").lower()

        if subdomain_strategy == "uniform":
            subdomains_list = generate_subdomains(
                domain=domain,
                overlap=cfg["overlap"],
                n_sub_per_dim=cfg["n_sub_per_dim"],
            )
            fixed_tw = cfg.get("fixed_transition_width", cfg["overlap"])

        elif subdomain_strategy == "zeros":
            # Non-uniform decomposition (feature aligned)
            subdomains_list = generate_subdomains_zeros(
                domain=domain,
                n_zeros_per_dim=cfg.get("n_zeros_per_dim", 11),  # 11 zeros → 10 subdomains
                overlap_abs=cfg.get("overlap_abs", None),
            )
            fixed_tw = cfg.get(
                "fixed_transition_width",
                cfg.get("overlap_abs", cfg.get("overlap", 0.0)),
            )

        else:
            raise ValueError(f"Unknown subdomain_strategy: {subdomain_strategy}")

        model = FBPINN(
            key=key,
            subdomains_list=subdomains_list,
            mlp_config=mlp_conf,
            ansatz=ansatz,
            fixed_transition_width=fixed_tw,
        )
    else:
        lo, hi = problem.domain
        model = PINN(key=key, ansatz=ansatz, mlp_config=mlp_conf, domain=(lo, hi))

    # Collocation and test points
    colloc = generate_collocation(
        domain, cfg["n_pts_per_dim"], strategy="uniform", seed=cfg["seed"]
    )
    if xdim == 1:
        lo = float(domain[0][0])
        hi = float(domain[1][0])
        x_test = jnp.linspace(lo, hi, 1000).reshape(-1, 1)
    else:
        x_test = generate_collocation(domain, cfg["n_test_pts_per_dim"], "grid")

    # Training
    if model_type == "fbpinn":
        model, loss_hist, (eval_steps, rel_l2_hist) = train_fbpinn(
            key=key,
            model=model,
            problem=problem,
            colloc=colloc,
            lr=cfg["lr"],
            steps=cfg["steps"],
            x_test=x_test,
            u_exact=u_exact,
            rad_cfg=cfg.get("rad_cfg"),
            eval_every=cfg.get("eval_every", 100),
        )
    else:
        model, loss_hist, (eval_steps, rel_l2_hist) = train_pinn(
            key=key,
            problem=problem,
            model=model,
            colloc=colloc,
            lr=cfg["lr"],
            steps=cfg["steps"],
            x_test=x_test,
            u_exact=u_exact,
            rad_cfg=cfg.get("rad_cfg"),
            eval_every=cfg.get("eval_every", 100),
        )

    # Visualization and metrics
    print("Training finished. Visualizing results and saving stats...")
    if xdim == 1:
        u_pred = model(x_test).squeeze()
        u_true = u_exact(x_test).squeeze()
        err = u_pred - u_true
        final_metrics = {
            "relative_l2_error": jnp.linalg.norm(err) / (jnp.linalg.norm(u_true) + 1e-8),
            "mae": jnp.mean(jnp.abs(err)),
            "mse": jnp.mean(err**2),
            "rmse": jnp.sqrt(jnp.mean(err**2)),
        }
        visualize_1d(
            x_test.squeeze(),
            u_true,
            u_pred,
            loss_hist,
            eval_steps,
            rel_l2_hist,
            save_dir=save_dir,
        )
        save_training_stats(loss_hist, eval_steps, rel_l2_hist, final_metrics, save_dir)
    else:
        n = cfg["n_test_pts_per_dim"]
        gx = jnp.linspace(domain[0][0], domain[1][0], n)
        gy = jnp.linspace(domain[0][1], domain[1][1], n)
        u_p = jax.vmap(model)(x_test).squeeze().reshape(n, n)
        u_e = u_exact(x_test).squeeze().reshape(n, n)
        _, final_metrics = visualize_2d(
            model, gx, gy, u_p, u_e, loss_hist, eval_steps, rel_l2_hist, save_dir=save_dir
        )

    # Plot subdomains only for FBPINN in 2D
    if model_type == "fbpinn" and xdim == 2:
        sub_arr = [jnp.stack(sd, 0) for sd in subdomains_list]
        plot_subdomains(sub_arr, save_path=os.path.join(save_dir, "subdomains.png"))

    print(f"All results saved in {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    cfg = {
        # Switch between CosineODE and FirstOrderFreq1010 as needed
        "pde_module": FirstOrderFreq1010,
        "model_type": "pinn",              # "pinn" or "fbpinn"

        # Training
        "lr": 1e-3,
        "steps": 300,
        "seed": 0,
        "eval_every": 100,

        # Data
        "n_pts_per_dim": 80,
        "n_test_pts_per_dim": 50,

        # FBPINN options (used only if model_type == "fbpinn")
        "subdomain_strategy": "uniform",     # "uniform" or "zeros"
        "n_zeros_per_dim": 11,             # only for feature aligned example
        "n_sub_per_dim": [10, 10],           # for the uniform strategy
        "overlap": 0.06,                    # for the uniform strategy
        "overlap_abs": 0.06,                # recommended for the zeros strategy

        # Network (also subdomains network for fbpinn)
        "width_size": 64,
        "depth": 2,

        # RAD (optional; handled inside trainers)
        "rad_cfg": None,
    }
    main(cfg)
