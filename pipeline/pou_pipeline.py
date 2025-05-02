import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as onp
import os

from train.trainer_single import train_single
from train.train_pou_rbf import train_pou_rbf
from model.rbf_pou import init_rbf_params, rbf_forward
from train.trainer_fbpinn import train_fbpinn
from utils.visualizer import plot_solution, plot_loss_curve, plot_pou_weights


def run_pipeline(config, pde_problem):
    output_dir = config["output_dir"]

    # === Phase A: training SinglePINN ===
    print("=== Phase A: Training SinglePINN ===")
    pinn_model = train_single(
        pde_problem=pde_problem,
        tol=config["pinn"]["tol"],
        max_steps=config["pinn"]["max_steps"]
    )

    # === visualize Phase A  ===
    x_eval = jnp.linspace(pde_problem["domain"][0], pde_problem["domain"][1], 300)
    u_pred = jax.vmap(pinn_model)(x_eval)
    u_true = pde_problem["u_exact"](x_eval)
    plot_solution(x_eval, u_pred, u_true, output_dir, model_name="SinglePINN", filename="phaseA_singlepinn.png")

    # === 用 PINN 解作为训练数据，用于 POU 网络训练 ===
    x_data = onp.linspace(pde_problem["domain"][0], pde_problem["domain"][1], 500)
    u_data = onp.array([float(pinn_model(jnp.array(x))) for x in x_data])

    # === Phase B: RBF-POU 训练 ===
    print("=== Phase B: Training RBF-POU ===")
    rng_pou = jr.PRNGKey(config["seed"])
    num_partitions = config["pou"]["num_partitions"]

    params_init = init_rbf_params(rng_pou, num_partitions)

    p1, p2 = train_pou_rbf(
        params_init, x_data, u_data,
        num_partitions=num_partitions,
        lambda_reg=config["pou"]["lambda_reg"],
        lr_phase1=config["pou"]["lr_phase1"],
        lr_phase2=config["pou"]["lr_phase2"],
        num_epochs_phase1=config["pou"]["num_epochs_phase1"],
        num_epochs_phase2=config["pou"]["num_epochs_phase2"]
    )

    # === 可视化 POU 权重函数 ===
    x_plot = jnp.linspace(pde_problem["domain"][0], pde_problem["domain"][1], 300)
    weights = rbf_forward(p2, x_plot)
    plot_pou_weights(x_plot, onp.array(weights), output_dir, filename="phaseB_pou_weights.png")

    # === 构建 window 函数（RBF输出） ===
    def partition_func(x):
        x_arr = jnp.atleast_1d(x)
        part = rbf_forward(p2, x_arr)
        if part.ndim == 2 and part.shape[0] == 1:
            return part[0]
        return part

    # === Phase C: FBPINN 训练（使用 POU 窗口） ===
    print("=== Phase C: Training FBPINN with POU window ===")
    fbpinn_model, loss_hist = train_fbpinn(
        pde_problem=pde_problem,
        window_fn=partition_func,
        n_sub=num_partitions,
        steps=config["fbpinn"]["train_steps"],
        lr=config["fbpinn"]["learning_rate"]
    )

    # === 可视化 FBPINN loss 和预测结果 ===
    plot_loss_curve(loss_hist, output_dir, title="FBPINN Loss", filename="phaseC_fbpinn_loss.png")

    u_fbpinn = jax.vmap(fbpinn_model)(x_eval)
    plot_solution(x_eval, u_fbpinn, u_true, output_dir, model_name="FBPINN", filename="phaseC_fbpinn_solution.png")

    return {
        "pinn_model": pinn_model,
        "pou_params": p2,
        "fbpinn_model": fbpinn_model,
        "loss_hist": loss_hist
    }
