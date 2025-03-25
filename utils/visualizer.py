# utils/visualizer.py
import os
import matplotlib.pyplot as plt
import jax.numpy as jnp

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
