from __future__ import annotations
import jax, jax.numpy as jnp, optax, matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)

# ------------------------------------------------------------------
# toy function (1-D)
# ------------------------------------------------------------------
def toy_func(x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.asarray(x)
    return jnp.exp(jnp.sin(0.3 * jnp.pi * (x**2)))

# ------------------------------------------------------------------
# MLP-PoU (same as 2-D, but input_dim=1)
# ------------------------------------------------------------------
def glorot(k, shape):
    fan_in, fan_out = shape
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(k, shape, minval=-limit, maxval=limit)

# ------------------------------------------------------------------
#  ResNet-PoU  (1-D, C experts, 2 residual blocks)
# ------------------------------------------------------------------
def glorot(key, shape):
    fan_in, fan_out = shape
    lim = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-lim, maxval=lim)

class ResNetPOUNet:
    """
    x ──► (FC→ReLU→FC + skip)×2 ─► FCout ─► softmax(C)
    """
    def __init__(self, input_dim: int, num_experts: int,
                 hidden_dim: int = 32, n_blocks: int = 2,
                 key: jax.random.KeyArray | None = None):

        key = jax.random.PRNGKey(0) if key is None else key
        keys = jax.random.split(key, n_blocks * 2 + 1)  # 2 matrices per block + out

        p = {}
        in_dim = input_dim
        for bi in range(n_blocks):
            # block i : W1,b1 , W2,b2
            k1, k2 = keys[2 * bi], keys[2 * bi + 1]
            p[f"W{bi}_1"] = glorot(k1, (in_dim, hidden_dim))
            p[f"b{bi}_1"] = jnp.zeros((hidden_dim,))
            p[f"W{bi}_2"] = glorot(k2, (hidden_dim, in_dim))   # back to in_dim for skip-add
            p[f"b{bi}_2"] = jnp.zeros((in_dim,))
        # output layer
        p["W_out"] = glorot(keys[-1], (in_dim, num_experts))
        p["b_out"] = jnp.zeros((num_experts,))
        self._init = p
        self.C = num_experts
        self.n_blocks = n_blocks
        self.in_dim = in_dim

    def init_params(self):
        return {k: v.copy() for k, v in self._init.items()}

    @staticmethod
    def _block(x, W1, b1, W2, b2):
        h = jax.nn.relu(x @ W1 + b1)
        h = h @ W2 + b2
        return x + h                    # residual add

    def forward(self, params: dict[str, jnp.ndarray], x: jnp.ndarray):
        h = x                           # x shape (N,1)
        for bi in range(self.n_blocks):
            h = self._block(h,
                             params[f"W{bi}_1"], params[f"b{bi}_1"],
                             params[f"W{bi}_2"], params[f"b{bi}_2"])
        logits = h @ params["W_out"] + params["b_out"]
        return jax.nn.softmax(logits, axis=-1)     # (N,C)


# ------------------------------------------------------------------
# local quadratic (1, x, x²)  → dim = 3
# ------------------------------------------------------------------
def _design_matrix(x: jnp.ndarray)->jnp.ndarray:      # x shape (N,1)
    x1 = x.squeeze(-1)
    return jnp.stack([jnp.ones_like(x1), x1, x1**2], -1)   # (N,3)

def fit_local_quad(x,y,w, lam:float=0.0):
    A, y = _design_matrix(x), y[:,None]
    def solve(weights):
        Aw = A * weights[:,None]
        M  = A.T @ Aw
        b  = (Aw.T @ y).squeeze(-1)
        return jnp.linalg.solve(M + lam*jnp.eye(3), b)
    return jax.vmap(solve,1,0)(w)                     # (C,3)

def predict_from_coeffs(x, coeffs, part):
    A = _design_matrix(x)
    y_cent = A @ coeffs.T      # (N,C)
    return jnp.sum(part * y_cent, 1)

# ------------------------------------------------------------------
# visualisation (1-D curves)
# ------------------------------------------------------------------
import pathlib, matplotlib.pyplot as plt
_SAVE_DIR = pathlib.Path("viz")         # 统一保存目录
_SAVE_DIR.mkdir(exist_ok=True)

def vis_partitions(model, params, title, grid=400, fname=None):
    xs = jnp.linspace(0, 6, grid)[:, None]  # Updated domain from [0, 1] to [0, 6]
    p   = model.forward(params, xs)
    plt.figure(figsize=(6, 3))
    for i in range(p.shape[1]):
        plt.plot(xs, p[:, i], label=f"p{i}")
    plt.ylim(0, 1); plt.title(title); plt.legend(); plt.tight_layout()
    if fname is None:
        fname = _SAVE_DIR / f"{title.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=300); plt.close()

def vis_final(model, params, x_train, y_train, coeffs, grid=400):
    xs = jnp.linspace(0, 6, grid)[:, None]  # Updated domain from [0, 1] to [0, 6]
    part = model.forward(params, xs)
    y_pred = predict_from_coeffs(xs, coeffs, part)
    y_true = toy_func(xs.squeeze(-1))
    plt.figure(figsize=(6, 3))
    plt.plot(xs, y_true, '--', label='truth')
    plt.plot(xs, y_pred, label='pred')
    plt.scatter(x_train, y_train, s=8, c='k', alpha=.3)
    plt.legend(); plt.tight_layout()
    fname = _SAVE_DIR / "final_prediction.png"
    plt.savefig(fname, dpi=300); plt.close()
    print("saved:", fname)

# ------------------------------------------------------------------
# internal 1-phase LSGD
# ------------------------------------------------------------------
def _run_lsgd(model, params, x, y,
              n_epochs, lr, lam_init, rho, n_stag,
              viz_interval=None, prints=10, start_ep=0):
    lam = lam_init; best, stag = jnp.inf, 0
    log_int = max(1, n_epochs//prints)

    @jax.jit
    def loss_fn(p, lamv):
        part   = model.forward(p, x)
        coeffs = fit_local_quad(x, y, part, lamv)
        pred   = predict_from_coeffs(x, coeffs, part)
        return jnp.mean((pred - y)**2)

    valgrad = jax.jit(lambda p,lv: jax.value_and_grad(lambda pp:loss_fn(pp,lv))(p))
    opt = optax.adam(lr); state = opt.init(params)
    g_ep = start_ep
    for epoch in range(n_epochs):
        loss, grads = valgrad(params, lam)
        updates, state = opt.update(grads, state); params = optax.apply_updates(params, updates)
        loss.block_until_ready()
        
        # Print the loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
        
        if viz_interval and epoch % viz_interval == 0:
            vis_partitions(model, params, title=f"epoch_{epoch:05d}",
                           fname=_SAVE_DIR / f"part_{epoch:05d}.png")
        if loss < best - 1e-12: best, stag = loss, 0
        else: stag += 1
        if stag > n_stag: lam *= rho; stag = 0
        g_ep += 1
    return params, g_ep

# ------------------------------------------------------------------
# two-phase trainer
# ------------------------------------------------------------------
def train_two_phase(model, x_train, y_train,
                    n_pre=4000, n_post=1000,
                    lr_pre=1e-3, lr_post=5e-4,
                    lam_init=1e-3, rho=0.99, n_stag=200,
                    viz_interval=500, prints=10):
    params = model.init_params()
    params, ep = _run_lsgd(model, params, x_train, y_train,
                           n_epochs=n_pre, lr=lr_pre,
                           lam_init=lam_init, rho=rho, n_stag=n_stag,
                           viz_interval=viz_interval, prints=prints, start_ep=0)
    params, _  = _run_lsgd(model, params, x_train, y_train,
                           n_epochs=n_post, lr=lr_post,
                           lam_init=0.0, rho=1.0, n_stag=n_stag,
                           viz_interval=viz_interval, prints=prints, start_ep=ep)
    part   = model.forward(params, x_train)
    coeffs = fit_local_quad(x_train, y_train, part, lam=0.0)
    return params, coeffs

# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    model = ResNetPOUNet(input_dim=1, num_experts=6, hidden_dim=32, n_blocks=2, key=key)
    x_train = jnp.linspace(0,6,200)[:,None]        # Updated domain from [0, 1] to [0, 6]
    y_train = toy_func(x_train.squeeze(-1))

    params, coeffs = train_two_phase(
        model, x_train, y_train,
        n_pre=10000, n_post=5000,
        lr_pre=1e-3, lr_post=5e-4,
        lam_init=1e-3, rho=0.99, n_stag=200,
        viz_interval=1000, prints=10
    )

    vis_partitions(model, params, title="final partitions")
    vis_final(model, params, x_train, y_train, coeffs)
