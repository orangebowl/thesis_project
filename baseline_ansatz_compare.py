import os  
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import matplotlib.pyplot as plt

print(jax.devices())

# ------------------------------
# Hyperparameters
# ------------------------------
N_COLLOCATION_POINTS = 1000  
LEARNING_RATE = 1e-3  
N_OPTIMIZATION_EPOCHS = 200000  

key = jr.PRNGKey(42)  

# ------------------------------
# PDE & Exact Solution
# ------------------------------
def phi(x):
    return (jnp.pi / 4.0) * x**2

def u_exact(x):
    return jnp.sin(phi(x))

def f_pde(x):
    return (jnp.pi**2 / 4.0) * x**2 * jnp.sin(phi(x)) - (jnp.pi / 2.0) * jnp.cos(phi(x))

# ------------------------------
# Ansatz Solution
# ------------------------------
def ansatz(x, nn_output):
    """
    Ansatz function ensures the boundary conditions are automatically satisfied.
    """
    A_x = (1 - jnp.exp(-x)) * (1 - jnp.exp(-(8 - x)))  # Ansatz transformation
    return A_x * nn_output

# ------------------------------
# Define PINN Model
# ------------------------------
class PINN(eqx.Module):
    mlp: eqx.nn.MLP
    use_ansatz: bool

    def __init__(self, key, use_ansatz=True):
        self.mlp = eqx.nn.MLP(
            in_size=1, out_size=1, width_size=20, depth=3,
            activation=jax.nn.tanh, key=key
        )
        self.use_ansatz = use_ansatz

    def __call__(self, x):
        nn_out = self.mlp(x[jnp.newaxis])[0]
        return ansatz(x, nn_out) if self.use_ansatz else nn_out

# Compute PDE residual
def pde_residual(params, x):
    u = lambda xx: params(xx)
    dudx = jax.grad(u)
    d2udx2 = jax.grad(dudx)
    return d2udx2(x) + f_pde(x)

# Loss function
def loss_fn(params, x_collocation, use_ansatz):
    loss_pde = jnp.mean(jax.vmap(lambda xx: pde_residual(params, xx)**2)(x_collocation))
    if use_ansatz:
        return loss_pde  # 边界条件由 Ansatz 解决，无需 loss_bc
    else:
        loss_bc = (params(jnp.array(0.0)) - 0)**2 + (params(jnp.array(8.0)) - 0)**2
        return loss_pde + loss_bc

# ------------------------------
# Training Function
# ------------------------------
def train_pinn(use_ansatz=True):
    model = PINN(key, use_ansatz)
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Collocation Points
    sampling_key = jr.PRNGKey(999)
    x_collocation = jr.uniform(sampling_key, (N_COLLOCATION_POINTS,), minval=0, maxval=8)

    # Training Step
    @eqx.filter_jit
    def train_step(params, opt_state, x_collocation):
        loss_val, grads = eqx.filter_value_and_grad(lambda p: loss_fn(p, x_collocation, use_ansatz))(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return eqx.apply_updates(params, updates), opt_state, loss_val

    # Training Loop
    loss_history = []
    for it in range(N_OPTIMIZATION_EPOCHS):
        model, opt_state, loss_val = train_step(model, opt_state, x_collocation)
        
        if it % 2000 == 0:
            print(f"Iter={it}, loss={loss_val}, Ansatz={use_ansatz}")
        loss_history.append(loss_val)

    return model, loss_history

# ------------------------------
# Train Two Models (With and Without Ansatz)
# ------------------------------
print("Training PINN with Ansatz...")
model_ansatz, loss_history_ansatz = train_pinn(use_ansatz=True)

print("Training PINN without Ansatz...")
model_no_ansatz, loss_history_no_ansatz = train_pinn(use_ansatz=False)

# ------------------------------
# Visualization
# ------------------------------
x_plot = jnp.linspace(0.0, 8.0, 200)  
u_pred_ansatz = jax.vmap(lambda x: model_ansatz(x))(x_plot)
u_pred_no_ansatz = jax.vmap(lambda x: model_no_ansatz(x))(x_plot)
u_true = u_exact(x_plot)

# Compute L1 Error
l1_error_ansatz = jnp.mean(jnp.abs(u_pred_ansatz - u_true))
l1_error_no_ansatz = jnp.mean(jnp.abs(u_pred_no_ansatz - u_true))
print(f"L1 Error (with Ansatz): {l1_error_ansatz:.6f}")
print(f"L1 Error (without Ansatz): {l1_error_no_ansatz:.6f}")

output_dir = "./figures_pinn"
os.makedirs(output_dir, exist_ok=True)

# (1) Predict vs Exact Solution (With and Without Ansatz)
plt.figure(figsize=(8, 5))
plt.plot(x_plot, u_true, "--", label="Exact Solution")
plt.plot(x_plot, u_pred_ansatz, label="PINN Solution (with Ansatz)")
plt.plot(x_plot, u_pred_no_ansatz, label="PINN Solution (without Ansatz)", linestyle="dotted")
plt.legend()
plt.grid(True)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title(f"1D PDE Solution - L1 Error\nAnsatz: {l1_error_ansatz:.6f}, No Ansatz: {l1_error_no_ansatz:.6f}")
plt.savefig(os.path.join(output_dir, "solution_comparison.png"), dpi=300)
plt.close()

# (2) Training Loss Comparison
plt.figure()
plt.plot(loss_history_ansatz, label="Loss with Ansatz")
plt.plot(loss_history_no_ansatz, label="Loss without Ansatz", linestyle="dashed")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Comparison")
plt.legend()
plt.savefig(os.path.join(output_dir, "training_loss_comparison.png"), dpi=300)
plt.close()

print("Done, saved the figures")
