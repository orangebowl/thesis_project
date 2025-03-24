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
# Define Neural Network with Ansatz
# ------------------------------
class PINN(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key):
        self.mlp = eqx.nn.MLP(
            in_size=1, out_size=1, width_size=20, depth=3,
            activation=jax.nn.tanh, key=key
        )

    def __call__(self, x):
        nn_out = self.mlp(x[jnp.newaxis])[0]
        return ansatz(x, nn_out)  # Apply Ansatz transformation

# Compute PDE residual
def pde_residual(params, x):
    u = lambda xx: params(xx)
    dudx = jax.grad(u)
    d2udx2 = jax.grad(dudx)
    return d2udx2(x) + f_pde(x)

# Loss function (only considers PDE residual since BCs are handled by Ansatz)
def loss_fn(params, x_collocation):
    loss_pde = jnp.mean(jax.vmap(lambda xx: pde_residual(params, xx)**2)(x_collocation))
    return loss_pde  # No explicit loss for BCs as they are automatically satisfied

# ------------------------------
# Training Initialization
# ------------------------------
model = PINN(key)
optimizer = optax.adam(LEARNING_RATE)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

# Collocation Points (Updated to [0,8])
sampling_key = jr.PRNGKey(999)
x_collocation = jr.uniform(sampling_key, (N_COLLOCATION_POINTS,), minval=0, maxval=8)

# Training Step
@eqx.filter_jit
def train_step(params, opt_state, x_collocation):
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(params, x_collocation)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return eqx.apply_updates(params, updates), opt_state, loss_val

# Training Loop
loss_history = []
for it in range(N_OPTIMIZATION_EPOCHS):
    model, opt_state, loss_val = train_step(model, opt_state, x_collocation)
    
    if it % 2000 == 0:
        print(f"Iter={it}, loss={loss_val}")
    loss_history.append(loss_val)

# ------------------------------
# Visualization
# ------------------------------
x_plot = jnp.linspace(0.0, 8.0, 200)  
u_pred = jax.vmap(lambda x: model(x))(x_plot)
u_true = u_exact(x_plot)

# Compute L1 Error
l1_error = jnp.mean(jnp.abs(u_pred - u_true))
print(f"L1 Error: {l1_error:.6f}")

output_dir = "./figures_pinn"
os.makedirs(output_dir, exist_ok=True)

# (1) Predict vs Exact Solution
plt.figure(figsize=(8, 5))
plt.plot(x_plot, u_true, "--", label="Exact Solution")
plt.plot(x_plot, u_pred, label="PINN Solution")
plt.legend()
plt.grid(True)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title(f"1D PDE Solution - L1 Error: {l1_error:.6f}")
plt.savefig(os.path.join(output_dir, "solution.png"), dpi=300)
plt.close()

# (2) Training Loss
plt.figure()
plt.plot(loss_history, label="Training Loss")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.savefig(os.path.join(output_dir, "training_loss.png"), dpi=300)
plt.close()

print("Done, saved the figure")
