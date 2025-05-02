# main.py ────────────────────────────────────────────────
"""
统一入口：读取 YAML 配置，构建问题 → 构建模型 → 训练 → 可视化
用法:
    python main.py --cfg config/run.yaml
"""
import argparse, importlib, yaml
from pathlib import Path
import jax, jax.numpy as jnp

# ────────────────────── CLI ──────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--cfg", default="config/run.yaml", help="YAML 配置文件路径")
CFG = yaml.safe_load(Path(ap.parse_args().cfg).read_text())

# ─────────────── 加载 PDEProblem 子类 ─────────────
mod_path, cls_name = CFG["pde"].rsplit(".", 1)
ProbCls = getattr(importlib.import_module(f"physics.{mod_path}"), cls_name)
prob      = ProbCls()
DOMAIN    = prob.domain           # (x_min, x_max)
RESIDUAL  = prob.residual         # PDE 残差函数

# ─────────────── MLP 配置字典 ───────────────
act_map = {"tanh": jax.nn.tanh, "relu": jax.nn.relu,
           "gelu": jax.nn.gelu, "sigmoid": jax.nn.sigmoid}
mlp_cfg = dict(CFG["mlp"])
mlp_cfg["activation"] = act_map[mlp_cfg["activation"]]

key = jax.random.PRNGKey(0)

# ─────────────── 构建模型 & collocation ───────────────
if CFG["model_type"].lower() == "fbpinn":
    from model.fbpinn_model import FBPINN
    from utils.data_utils   import generate_subdomain, generate_collocation_points

    n_sub   = CFG["training"]["n_sub"]  # number of subdomains
    overlap = CFG["training"]["overlap"] # param. for overlap of window functions
    subs    = generate_subdomain(DOMAIN, n_sub, overlap) # generate subdomains

    model   = FBPINN(key, n_sub, prob.ansatz, subs, mlp_cfg) 

    n_pts   = CFG["training"]["n_points_per_subdomain"]
    colloc, _ = generate_collocation_points(
        domain=DOMAIN, subdomains_list=subs,
        n_points_per_subdomain=n_pts, seed=0
    )                                       # list[n_sub] -> full-batch
    trainer_mod = "train.trainer_fbpinn"
else:
    from model.pinn_model import PINN
    model = PINN(key, prob.ansatz,
                 width=mlp_cfg["width_size"], depth=mlp_cfg["depth"])

    total   = CFG["training"]["n_sub"] * CFG["training"]["n_points_per_subdomain"]
    colloc  = jnp.linspace(*DOMAIN, total)             # (N,)
    trainer_mod = "train.trainer_single"

# ─────────────── 训练 ───────────────
TrainMod = importlib.import_module(trainer_mod)
train_fn = getattr(TrainMod, "train_fbpinn" if trainer_mod.endswith("fbpinn")
                                else "train_single")

steps, lr = CFG["training"]["steps"], CFG["training"]["lr"]
out_dir   = Path(CFG["save"]["output_dir"]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

if trainer_mod.endswith("fbpinn"):
    # FBPINN 
    model, loss_hist, (t_steps, t_l1) = train_fn(
        model                       = model,
        subdomain_collocation_points= colloc,
        steps                       = steps,
        lr                          = lr,
        pde_residual_loss           = RESIDUAL,
        x_test                      = jnp.linspace(*DOMAIN, 300),
        u_exact                     = prob.exact,
        save_dir                    = str(out_dir),
        checkpoint_every            = CFG["save"].get("checkpoint_every", 0)
    )
else:
    # PINN –> 仍然可 mini-batch
    model, loss_hist, t_l1 = train_fn(
        model, colloc, lr, steps,
        RESIDUAL, jnp.linspace(*DOMAIN, 300), prob.exact,
        batch_size=CFG["training"]["batch_size"]
    )
    t_steps = jnp.arange(len(t_l1))

# ─────────────── 可视化 & 保存 ───────────────
from utils.visualizer import (
    plot_prediction_vs_exact, plot_training_loss, plot_test_l1_curve,
    plot_window_weights, plot_subdomain_partials, save_training_stats)

x = jnp.linspace(*DOMAIN, 500)
u_pred, u_true = jax.vmap(model)(x), prob.exact(x)

plot_prediction_vs_exact(x, u_true, u_pred, out_dir)
plot_training_loss(loss_hist, out_dir)
plot_test_l1_curve(t_steps, t_l1, out_dir)

if trainer_mod.endswith("fbpinn"):
    plot_window_weights(x, subs, len(subs), out_dir)
    plot_subdomain_partials(model, x, u_true, out_dir)

save_training_stats(loss_hist, t_steps, t_l1, out_dir)
print(f"✓ Done. Results saved to {out_dir}")
