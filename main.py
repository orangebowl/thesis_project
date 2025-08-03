import os
import datetime
import jax
import jax.numpy as jnp
import itertools
from typing import Dict, Any, List, Union

# --- Local Imports ---
# Added ViscousBurgersFBPINN to the imports
from utils.data_utils import generate_subdomains, generate_collocation
from physics.problems import Poisson2D_freq, CosineODE, FirstOrderFreq1010,ViscousBurgersFBPINN
from model.fbpinn_model import FBPINN
from model.pinn_model import PINN
from train.trainer_fbpinn import train_fbpinn
from train.trainer_single import train_single
from vis.vis_1d import visualize_1d
from vis.vis_2d import visualize_2d,plot_subdomains

def main(cfg: Dict[str, Any]):
    """
    Main function to configure and run the PINN/FBPINN experiment.
    """
    # --- 1. Setup & Problem Definition ---
    SAVE_DIR = os.path.join("./ckpts/", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Results will be saved in: {os.path.abspath(SAVE_DIR)}")

    pde_module = cfg["pde_module"]
    problem = pde_module()
    u_exact, ansatz, domain, residual_fn = problem.exact, problem.ansatz, problem.domain, problem.residual

    # Determine problem dimension from domain shape
    try:
        xdim = domain[0].size
    except (AttributeError, IndexError):
        xdim = 1
    print(f"Problem Dimension: {xdim}D")

    key = jax.random.PRNGKey(cfg["seed"])

    # --- 2. Model Initialization ---
    mlp_conf = dict(
        in_size=xdim, 
        out_size=1,
        width_size=cfg["width_size"], 
        depth=cfg["depth"],
        activation=jax.nn.tanh
    )

    model_type = cfg["model_type"].lower()
    if model_type == "fbpinn":
        # --- Subdomain Generation using the provided utility function ---
        print(f"Generating subdomains with config: n_sub_per_dim={cfg['n_sub_per_dim']}, overlap={cfg['overlap']}")
        
        # Directly call the robust utility function provided
        # It handles integer and list inputs for n_sub_per_dim internally.
        subdomains_list = generate_subdomains(
            domain=domain,
            overlap=cfg["overlap"],
            n_sub_per_dim=cfg["n_sub_per_dim"]
        )

        print(f"Generated {len(subdomains_list)} total subdomains.")
        
        # Note: The FBPINN initialization signature is kept consistent with the robust version.
        model = FBPINN(
            key=key,
            subdomains_list=subdomains_list,
            mlp_config=mlp_conf,
            ansatz=ansatz,
            fixed_transition_width=cfg["overlap"],
        )
        '''
        # ---------- 0. 参数 ----------
        overlap_abs = cfg["overlap_abs"]   # 例如 0.08
        nx, ny      = cfg["nx"], cfg["ny"] # 10 & 10  → 100 blocks

        # ---------- 1. 零点 → 中心 & 宽度(含交叠) ----------
        x_zeros = jnp.sqrt(jnp.arange(nx + 1) / nx)   # n=0..nx
        y_zeros = jnp.sqrt(jnp.arange(ny + 1) / ny)   # m=0..ny

        x_l, x_r   = x_zeros[:-1], x_zeros[1:]
        y_l, y_r   = y_zeros[:-1], y_zeros[1:]

        x_centers  = 0.5 * (x_l + x_r)
        y_centers  = 0.5 * (y_l + y_r)

        x_widths   = (x_r - x_l) + overlap_abs   # ★ 把交叠量加进来
        y_widths   = (y_r - y_l) + overlap_abs

        # ---------- 2. 组装 (lower, upper) ----------
        subdomains_list = []
        for xc, wx in zip(x_centers, x_widths):
            for yc, wy in zip(y_centers, y_widths):
                hx, hy = wx / 2.0, wy / 2.0

                lower = jnp.array([xc - hx, yc - hy])
                upper = jnp.array([xc + hx, yc + hy])

                subdomains_list.append((lower, upper))   # tuple, 与 generate_subdomains 输出一致

        print(f"Generated {len(subdomains_list)} sub-domains "
            f"({nx} × {ny}), width includes overlap={overlap_abs}")

        # ------------------ 2) 初始化 FBPINN ------------------
        model = FBPINN(
            key=key,
            subdomains_list=subdomains_list,
            mlp_config=mlp_conf,
            ansatz=ansatz,
            fixed_transition_width=overlap_abs,   # 与宽度里那 0.06 对应
        )'''
    elif model_type == "pinn":
        model = PINN(
            key=key,
            ansatz=ansatz,
            mlp_config=mlp_conf,
        )
    else:
        raise ValueError(f"Unknown model_type: {cfg['model_type']}")

    print(f"Initialized {model_type.upper()} model.")

    # --- 3. Data Generation ---
    num_colloc_total = cfg["n_pts_per_dim"] 
    global_collocation_points = generate_collocation(domain, num_colloc_total, strategy="grid")
    print(f"Generated {global_collocation_points.shape[0]} grid collocation points for training.")

    # Test points for validation and plotting
    if xdim == 1:
        x_test = jnp.linspace(domain[0], domain[1], 1000).reshape(-1, 1)
    else: # xdim == 2
        num_test_total = cfg["n_test_pts_per_dim"]
        x_test = generate_collocation(domain, num_test_total, strategy="grid")
    print(f"Generated {x_test.shape[0]} test points for validation.")

    # --- 4. Training ---
    # Note: The train_fbpinn signature is kept consistent with the robust version.
    if model_type == "fbpinn":
        model, loss_hist, (l1_steps, l1_hist) = train_fbpinn(
            key = key,
            model=model,
            problem=problem,
            colloc=global_collocation_points,
            steps=cfg["steps"],
            lr=cfg["lr"],
            x_test=x_test,
            u_exact=u_exact,
        )
    else: # pinn
        model, loss_hist, (l1_steps, l1_hist) = train_single(
            model=model,
            colloc=global_collocation_points,
            lr=cfg["lr"],
            steps=cfg["steps"],
            pde_residual=residual_fn,
            batch_size=cfg["batch_size"],
            x_test=x_test,
            u_exact=u_exact,
            save_dir=SAVE_DIR,
            checkpoint_every=0,
        )

    # --- 5. Visualization and Results ---
    print("Training finished. Saving plots...")
    if xdim == 1:
        u_pred = jax.vmap(model)(x_test).squeeze()
        u_true = u_exact(x_test).squeeze()
        plot_paths = visualize_1d(
            x_test.squeeze(), u_true, u_pred,
            loss_hist, l1_steps, l1_hist,
            save_dir=SAVE_DIR
        )
    else: # xdim == 2
        test_n = cfg["n_test_pts_per_dim"]
        grid_x = jnp.linspace(domain[0][0], domain[1][0], test_n)
        grid_y = jnp.linspace(domain[0][1], domain[1][1], test_n)
        
        u_pred_grid = jax.vmap(model)(x_test).squeeze().reshape(test_n, test_n).T
        u_exact_grid = u_exact(x_test).reshape(test_n, test_n)
        plot_paths = visualize_2d(
            model, grid_x, grid_y,
            u_pred_grid, u_exact_grid,
            loss_hist, l1_steps, l1_hist,
            save_dir=SAVE_DIR
        )

    final_l1 = float(l1_hist[-1]) if len(l1_hist) > 0 else float("nan")
    print(f"\nFinal L1 error = {final_l1:.4e}")
    print("Saved files:")
    for name, path in plot_paths.items():
        print(f"  {name:<20s}: {path}")
    
    if model_type == "fbpinn":
        subdomains_arr = [jnp.stack(sd, axis=0) for sd in subdomains_list]
        plot_subdomains(subdomains_arr, save_path=os.path.join(SAVE_DIR, "subdomains.png"))
        print(f"  {'subdomains':<20s}: {os.path.join(SAVE_DIR, 'subdomains.png')}")

if __name__ == "__main__":
    cfg = {
        "pde_module": FirstOrderFreq1010,
        #  "fbpinn" or "pinn"
        "model_type": "fbpinn",
        # Training parameters
        "lr": 1e-3,
        "steps": 30000,
        "seed": 0,
        # --- 不均匀子域参数 ---
        
        #"overlap_abs": 0.06,   # abs overalp
        #"nx": 10,               # x 方向子域数 (= 零点个数-1)
        #"ny": 10,               # y 方向子域数
        
        # Data generation parameters
        "n_pts_per_dim": 100,     # Number of collocation points per dimension for training
        "n_test_pts_per_dim": 100, # Number of test points per dimension for validation
        
        # FBPINN specific parameters
        "n_sub_per_dim": [10, 10], # Use a list for non-uniform division, e.g., [4, 2] for 4 subs in x, 2 in y.
        "overlap": 0.06,
        # PINN specific parameters, ignore for FBPINN
        "batch_size": 10000,
        # MLP architecture
        "width_size": 64,
        "depth": 2,
    }
    main(cfg)
    
