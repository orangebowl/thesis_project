import pathlib
import jax, jax.numpy as jnp, optax, equinox as eqx
from jax import random, jacfwd, jacrev, vmap
import matplotlib.pyplot as plt
from tqdm import trange

# 可视化输出目录
VIZ_DIR = pathlib.Path("viz")
VIZ_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# 1) ResPOUNet2D + Two-phase LSGD
# ----------------------------------------------------------------------
def box_init(rng, w_in, w_out, m, delta, L, l):
    k1, k2 = random.split(rng)
    p = m * random.uniform(k1, (w_in, w_out))
    n = random.normal(k2, (w_in, w_out))
    n = n / (jnp.linalg.norm(n, axis=0, keepdims=True) + 1e-9)
    p_max = m * jnp.maximum(0.0, jnp.sign(n))
    k = 1.0 / ((L - 1) * (jnp.sum((p_max - p) * n, axis=0) + 1e-9))
    return k*n, jnp.sum(k*n*p, axis=0)

def init_linear_layer(rng, d_in, d_out, idx, L):
    W, b = box_init(rng, d_in, d_out, (1 + 1 / (L - 1)) ** (idx + 1), 1 / L, L, idx + 1)
    return {"W": W, "b": b}

def apply_linear(p, x):
    return x @ p["W"] + p["b"]

class ResPOUNet2D:
    def __init__(self, input_dim, num_partitions, hidden_dim, depth, key=None):
        if key is None: key = random.PRNGKey(0)
        keys = random.split(key, depth)
        layers = [init_linear_layer(keys[0], input_dim, hidden_dim, 0, depth)]
        for i in range(1, depth - 1):
            layers.append(init_linear_layer(keys[i], hidden_dim, hidden_dim, i, depth))
        layers.append(init_linear_layer(keys[-1], hidden_dim, num_partitions, depth - 1, depth))
        self._init = layers
        self.C = num_partitions

    def init_params(self):
        return jax.tree_util.tree_map(lambda x: x.copy(), self._init)

    def forward(self, params, x):
        h = jax.nn.relu(apply_linear(params[0], x))
        for p in params[1:-1]:
            h = h + jax.nn.relu(apply_linear(p, h))
        logits = apply_linear(params[-1], h)
        return jax.nn.softmax(logits, axis=-1)

def toy_func(xy):
    x, y = xy[..., 0], xy[..., 1]
    return jnp.sin(2 * jnp.pi * x**2) * jnp.sin(2 * jnp.pi * y**2)

def _design(xy):
    x, y = xy[:, 0], xy[:, 1]
    return jnp.stack([jnp.ones_like(x), x, y, x**2, x*y, y**2], axis=-1)

def fit_local_poly2(xy, f, w, lam=0.0):
    A, y = _design(xy), f[:, None]
    def solve(ws):
        Aw = A * ws[:, None]
        M = A.T @ Aw
        b = (Aw.T @ y).squeeze(-1)
        return jnp.linalg.solve(M + lam * jnp.eye(6), b)
    return vmap(solve, in_axes=1, out_axes=0)(w)

def predict(xy, coeffs, part):
    return jnp.sum(part * (_design(xy) @ coeffs.T), axis=1)

def _run_lsgd(net, params, xy, f, epochs, lr, lam0, rho, n_stag):
    lam = jnp.array(lam0); best, stag = jnp.inf, 0
    @jax.jit
    def loss_fn(p, lam_):
        w = net.forward(p, xy)
        c = fit_local_poly2(xy, f, w, lam_)
        pred = predict(xy, c, w)
        return jnp.mean((pred - f)**2)
    valgrad = jax.jit(lambda p, lam_: jax.value_and_grad(lambda q: loss_fn(q, lam_))(p))
    opt = optax.adam(lr); state = opt.init(params)
    for _ in range(epochs):
        loss, grads = valgrad(params, lam)
        updates, state = opt.update(grads, state)
        params = optax.apply_updates(params, updates)
        if loss < best - 1e-12:
            best, stag = loss, 0
        else:
            stag += 1
        if stag > n_stag:
            lam *= rho; stag = 0
    return params

def train_two_phase(net, xy, f,
                    n_pre=4000, n_post=2000,
                    lr_pre=1e-3, lr_post=5e-4,
                    lam0=1e-3, rho=0.99, n_stag=100):
    p = net.init_params()
    p = _run_lsgd(net, p, xy, f, n_pre, lr_pre, lam0, rho, n_stag)
    p = _run_lsgd(net, p, xy, f, n_post, lr_post, 0.0, 1.0, n_stag)
    return p

# ----------------------------------------------------------------------
# 2) Poisson2D_freq
# ----------------------------------------------------------------------
class Poisson2D_freq:
    domain = (jnp.array([0.,0.]), jnp.array([1.,1.]))

    @staticmethod
    def exact(xy):
        x, y = xy[..., 0], xy[..., 1]
        return jnp.sin(2 * jnp.pi * x**2) * jnp.sin(2 * jnp.pi * y**2)

    @staticmethod
    def ansatz(xy, u):
        x, y = xy[..., 0], xy[..., 1]
        return x * (1 - x) * y * (1 - y) * u

    @staticmethod
    def rhs(xy):
        x, y = xy[..., 0], xy[..., 1]
        sx, cx = jnp.sin(2 * jnp.pi * x**2), jnp.cos(2 * jnp.pi * x**2)
        sy, cy = jnp.sin(2 * jnp.pi * y**2), jnp.cos(2 * jnp.pi * y**2)
        return -4 * jnp.pi * (sy * cx + sx * cy) + 16 * jnp.pi**2 * (x**2 + y**2) * sx * sy

    def _single_res(self, m, pts):
        def u_fn(p): return m(p.reshape(1, 2)).squeeze()
        H = vmap(jacfwd(jacrev(u_fn)))(pts)
        lap = jnp.trace(H, -2, -1)
        return jnp.mean((-lap - self.rhs(pts))**2)

    def residual(self, m, xy):
        if isinstance(xy, list):
            return sum(self._single_res(m, p) for p in xy)
        else:
            return self._single_res(m, xy)

# ----------------------------------------------------------------------
# 3) FBPINNWithWindow with normalization
# ----------------------------------------------------------------------
class FBPINNWithWindow(eqx.Module):
    subnets: tuple
    ansatz: callable = eqx.static_field()
    subdomains: tuple = eqx.static_field()
    num_subdomains: int = eqx.static_field()
    pou_net: ResPOUNet2D = eqx.static_field()
    pou_params: any = eqx.static_field()

    def __init__(self, key, n_sub, ansatz, subds, mlp_cfg, pou_net, pou_params):
        self.ansatz = ansatz
        self.subdomains = subds
        self.num_subdomains = n_sub
        object.__setattr__(self, "pou_net", pou_net)  # PoU network
        object.__setattr__(self, "pou_params", pou_params)  # PoU parameters
        ks = random.split(key, n_sub)
        self.subnets = tuple(
            eqx.nn.MLP(
                in_size=mlp_cfg["in_size"],
                out_size=mlp_cfg["out_size"],
                width_size=mlp_cfg["width_size"],
                depth=mlp_cfg["depth"],
                activation=mlp_cfg["activation"],
                key=k
            ) for k in ks
        )

    def subdomain_window(self, i, x):
        """
        Calculate the window function based on PoU network output, which is a trainable weight
        """
        x = jnp.atleast_2d(x)
        w = self.pou_net.forward(self.pou_params, x)  # Output of PoU network, weights for each subdomain
        return w[:, i]  # Return the weight for the i-th subdomain, which will be used in the final weighted sum

    def __call__(self, x):
        """
        Compute the ansatz value for the given input `x`, weighted by the subdomain windows
        """
        x = jnp.atleast_2d(x)
        acc = 0.0
        for j in range(self.num_subdomains):
            wj = self.subdomain_window(j, x)  # Get weight for each subdomain from PoU
            left, right = self.subdomains[j]  # Get subdomain bounds
            center = (left + right) / 2  # Normalize by center
            scale  = (right - left) / 2   # Normalize by scale
            x_norm = (x - center) / scale  # Normalize the input `x` for each subdomain
            uj = jax.vmap(self.subnets[j])(x_norm)[:, 0]  # Get prediction for each subdomain
            acc += wj * uj  # Accumulate weighted predictions
        return self.ansatz(x, acc)  # Return the ansatz value with the accumulated weighted predictions

# ----------------------------------------------------------------------
# 4) train_step + compute_l1
# ----------------------------------------------------------------------
def make_train_step(optm, pde):
    @eqx.filter_jit
    def step(m, st, coll):
        loss_fn = lambda mm: pde.residual(mm, coll)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(m)
        ups, st = optm.update(grads, st, m)
        m = eqx.apply_updates(m, ups)
        return m, st, loss
    return step

@eqx.filter_jit
def compute_l1(model, xt, ue):
    pred = jax.vmap(model)(xt).squeeze()
    return jnp.mean(jnp.abs(pred - ue.squeeze()))

# ----------------------------------------------------------------------
# 5) Main
# ----------------------------------------------------------------------
if __name__=="__main__":
    # 5.1 train PoU
    n_sub = 4
    pou = ResPOUNet2D(2, n_sub, 32, 6, random.PRNGKey(0))
    xs = jnp.linspace(0, 1, 200)
    G = jnp.stack(jnp.meshgrid(xs, xs), -1).reshape(-1, 2)
    p_params = train_two_phase(pou, G, toy_func(G),
                               n_pre=3000, n_post=1000,
                               lr_pre=1e-3, lr_post=5e-4,
                               lam0=1e-3, rho=0.99, n_stag=50)

    # 5.2 save learned partitions
    part = pou.forward(p_params, G).reshape(200, 200, n_sub)
    fig, axs = plt.subplots(1, n_sub, figsize=(4 * n_sub, 4))
    for j in range(n_sub):
        im = axs[j].imshow(part[:, :, j], origin="lower",
                           extent=[0, 1, 0, 1], vmin=0, vmax=1)
        axs[j].set_title(f"Partition {j}")
        axs[j].set_xticks([0, 0.5, 1]); axs[j].set_yticks([0, 0.5, 1])
        axs[j].set_xlabel("x"); axs[j].set_ylabel("y")
        fig.colorbar(im, ax=axs[j], label="weight")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "learned_partitions.png", dpi=150)
    plt.close(fig)

    # 5.3 prepare subdomains & collocation
    cuts = jnp.linspace(0, 1, 3)
    subs = [(jnp.array([cuts[i], cuts[j]]), jnp.array([cuts[i + 1], cuts[j + 1]]))
            for i in range(2) for j in range(2)]
    Xc = random.uniform(random.PRNGKey(1), (5000, 2))
    Wc = pou.forward(p_params, Xc)
    coll = [Xc[Wc[:, k] > 1e-6] for k in range(n_sub)]

    # 5.4 build & train FBPINN
    mlp_cfg = {"in_size": 2, "out_size": 1, "width_size": 64, "depth": 2, "activation": jax.nn.tanh}
    model = FBPINNWithWindow(random.PRNGKey(2), n_sub,
                             Poisson2D_freq.ansatz, subs,
                             mlp_cfg, pou, p_params)
    opt = optax.adam(1e-3)
    st = opt.init(eqx.filter(model, eqx.is_array))
    step_fn = make_train_step(opt, Poisson2D_freq())

    for it in trange(2000, desc="Train FBPINN"):
        model, st, loss = step_fn(model, st, coll)
        if it % 100 == 0 or it == 1999:
            print(f"Step {it:4d} | Loss = {loss:.3e}")

    # 5.5 save final prediction
    Up = model(G).reshape(200, 200)
    Ue = Poisson2D_freq.exact(G).reshape(200, 200)
    Err = jnp.abs(Up - Ue)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    im1 = ax1.imshow(Up, origin="lower", extent=[0, 1, 0, 1])
    ax1.set_title("Prediction"); ax1.set_xticks([0, 0.5, 1]); ax1.set_yticks([0, 0.5, 1])
    fig.colorbar(im1, ax=ax1, label="u")
    im2 = ax2.imshow(Err, origin="lower", extent=[0, 1, 0, 1], cmap="inferno")
    ax2.set_title("Absolute Error"); ax2.set_xticks([0, 0.5, 1]); ax2.set_yticks([0, 0.5, 1])
    fig.colorbar(im2, ax=ax2, label="|u - u_exact|")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "final_prediction.png", dpi=150)
    plt.close(fig)
