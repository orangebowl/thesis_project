# utils/visualizer.py
"""
Here we implement the visualization

1) Training loss 
2) L1-Test loss
3) Plot partial solutions of each subdomains
4) Compare the prediction with the exact solution if exact solution is not None
5) Compare the test loss of fbpinns with single pinn
6) Plot the window functions

"""
import os
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from utils.window_function import my_window_func

def plot_subdomain_partials(model, x_test, u_true, save_dir):
    #print("[DEBUG] plot_subdomain_partials is called.")
    total_with_ansatz = []

    for i in range(len(model.subnets)):
        partial_solution_i = jax.vmap(lambda x: model.subdomain_pred(i, x))(x_test)
        partial_solution_i = partial_solution_i.reshape(-1)

        window_i = model.subdomain_window(i, x_test)

        windowed_partial = window_i * partial_solution_i

        total_solution_i = model.ansatz(x_test, windowed_partial)
        total_with_ansatz.append(total_solution_i)

    total_with_ansatz = jnp.stack(total_with_ansatz, axis=1)

    plt.figure(figsize=(10, 6))
    n_sub = len(model.subnets)

    for i in range(n_sub):
        plt.plot(x_test, total_with_ansatz[:, i], label=f"Subdomain {i} Solution")

    plt.plot(x_test, u_true, label="Exact Solution", color='black', linestyle='--', linewidth=2)

    plt.xlabel('x')
    plt.ylabel('Solution')
    plt.title('Subdomain Solutions vs Exact Solution')
    plt.legend(fontsize=8)
    plt.grid(True)
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "subdomain_partials_with_exact.png"))
        plt.close()

def plot_prediction_vs_exact(x_test, u_true, u_pred, save_dir):
    plt.figure()
    plt.plot(x_test, u_pred, label="FBPINN Pred")
    plt.plot(x_test, u_true, "--", label="Exact")
    plt.title("FBPINN Prediction vs Exact")
    plt.legend()
    plt.grid(True)
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "fbpinn_prediction.png"), dpi=300)
        plt.close()


def plot_training_loss(train_loss, save_dir):
    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.yscale("log")
    plt.xlabel("Steps")
    plt.grid(True)
    plt.title("Training Loss")
    plt.legend()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "fbpinn_loss_curve.png"), dpi=300)
        plt.close()


def plot_test_l1_curve(test_steps, test_l1, save_dir):
    plt.figure()
    plt.plot(test_steps, test_l1, label="Test L1 Error")
    plt.yscale("log")
    plt.xlabel("Steps")
    plt.grid(True)
    plt.title("Test L1 Error Curve")
    plt.legend()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "fbpinn_test_l1_curve.png"), dpi=300)
        plt.close()


def plot_window_weights(x_test, subdomains,n_sub, save_dir):
    w_all = my_window_func(subdomains, n_sub, x_test)
    plt.figure()
    for i in range(w_all.shape[1]):
        plt.plot(x_test, w_all[:, i], label=f"Window {i}")
    plt.title("Window Function Weights")
    plt.legend()
    plt.grid(True)
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "window_weights.png"), dpi=300)
        plt.close()


def save_training_stats(train_loss, test_steps, test_l1, save_dir):
    jnp.savez(os.path.join(save_dir, "fbpinn_stats.npz"),
              train_loss=train_loss, test_steps=test_steps, test_l1=test_l1)
    print(f"FBPINNs finished. Results in {save_dir}")


def plot_loss_curve(loss_history, save_path=None, title="Training Loss"):
    plt.figure()
    plt.plot(loss_history, label="Loss")
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_loss_compare(loss_single, loss_fbpinn, save_path=None):
    plt.figure()
    plt.plot(loss_single, label="Single PINN")
    plt.plot(loss_fbpinn, label="FBPINN")
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_solution_compare(x, u_pinn, u_fbpinn, u_exact, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(x, u_exact, "--", label="Exact Solution")
    plt.plot(x, u_pinn, label="Single PINN")
    plt.plot(x, u_fbpinn, label="FBPINN")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("Solution Comparison")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_window_functions(x, subdomains, sigma, window_function, save_path=None):
    """
    用于绘制 FB-PINN 中各子域的窗口函数权重
    - x: jnp.linspace 生成的输入
    - subdomains: [(a, b), (b, c), ...]
    - sigma: 平滑度参数
    - window_function: 你在 model 中实现的 sigmoid_window_function
    """
    weights = window_function(x, jnp.array(subdomains), sigma)
    plt.figure()
    for i in range(weights.shape[1]):
        plt.plot(x, weights[:, i], label=f"Window {i+1}")
    plt.xlabel("x")
    plt.ylabel("Weight")
    plt.title("FBPINN Window Functions")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_loss_curve(loss_list, output_dir, title="Training Loss", filename="loss_curve.png"):
    plt.figure()
    plt.plot(loss_list, label="Loss")
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


def plot_solution(x, u_pred, u_true, output_dir, model_name="Model", filename="solution.png"):
    plt.figure()
    plt.plot(x, u_true, '--', label="Exact")
    plt.plot(x, u_pred, label=f"{model_name}")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid(True)
    plt.legend()
    plt.title(f"{model_name} Prediction vs Exact")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


def plot_pou_weights(x, partitions, output_dir, filename="pou_weights.png"):
    """
    partitions: shape=(N, num_partitions)
    """
    plt.figure()
    for i in range(partitions.shape[1]):
        plt.plot(x, partitions[:, i], label=f"w_{i+1}")
    plt.xlabel("x")
    plt.ylabel("Weight")
    plt.title("Partition of Unity Weights")
    plt.legend()
    plt.grid(True)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
