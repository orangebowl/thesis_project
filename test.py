import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from tqdm import trange
from typing import Callable


# ==============================================================
# 1. PDE 定义（FirstOrderFreq68）
# ==============================================================
class FirstOrderFreq68:
    domain = (jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))

    @staticmethod
    def ansatz(xy, nn_out):
        x, y = xy[..., 0], xy[..., 1]
        return (x * y)[..., None] * nn_out

    @staticmethod
    def rhs(xy):
        x, y = xy[..., 0], xy[..., 1]
        alpha, beta = 6 * jnp.pi, 8 * jnp.pi
        term_x = 12 * jnp.pi * x * jnp.cos(alpha * x**2) * jnp.sin(beta * y**2)
        term_y = 16 * jnp.pi * y * jnp.sin(alpha * x**2) * jnp.cos(beta * y**2)
        return term_x + term_y

    # --- PINN 残差 ---
    def _pointwise_res(self, model, xy_batch):
        xy_batch = jnp.atleast_2d(xy_batch)

        def u_fn(pt):
            return jnp.ravel(model(pt[None, :]))[0]  # 保证输出为标量，兼容JAX

        u_val, grad_u = jax.vmap(jax.value_and_grad(u_fn))(xy_batch)
        u_x, u_y = grad_u[:, 0], grad_u[:, 1]
        return (u_x + u_y) - self.rhs(xy_batch)

    def residual(self, model, xy):
        r = self._pointwise_res(model, xy)
        return jnp.mean(r**2)


# ==============================================================
# 2. FBPINN
# ==============================================================
class FBPINN(eqx.Module):
    subnet_arrays: ...
    subnet_static: ... = eqx.static_field()
    ansatz: Callable = eqx.static_field()
    xmins_all: jax.Array = eqx.static_field()
    xmaxs_all: jax.Array = eqx.static_field()
    w_mu_min: jax.Array = eqx.static_field()
    w_sd_min: jax.Array = eqx.static_field()
    w_mu_max: jax.Array = eqx.static_field()
    w_sd_max: jax.Array = eqx.static_field()
    _centres: jax.Array = eqx.static_field()
    _scales: jax.Array = eqx.static_field()
    eps: float = eqx.static_field()
    n_sub: int = eqx.static_field()

    def __init__(self, *, key, subdomains, mlp_cfg, ansatz,
                 fixed_transition=0.3, eps=1e-8):
        self.ansatz, self.eps = ansatz, float(eps)
        self.n_sub = len(subdomains)

        # 1) 子域
        self.xmins_all = jnp.stack([s[0] for s in subdomains])
        self.xmaxs_all = jnp.stack([s[1] for s in subdomains])

        # 2) window
        ft = fixed_transition
        w = jnp.full_like(self.xmins_all, ft)
        t = jnp.log((1 - self.eps) / self.eps)
        self.w_mu_min = self.xmins_all + w / 2
        self.w_sd_min = w / (2 * t)
        self.w_mu_max = self.xmaxs_all - w / 2
        self.w_sd_max = w / (2 * t)

        # 3) 子网
        nets = [
            eqx.nn.MLP(
                in_size=mlp_cfg["in_size"],
                out_size=mlp_cfg["out_size"],
                width_size=mlp_cfg["width_size"],
                depth=mlp_cfg["depth"],
                activation=jax.nn.tanh,
                final_activation=lambda x: x,
                key=k,
            )
            for k in jax.random.split(key, self.n_sub)
        ]
        arrays, statics = zip(*(eqx.partition(n, eqx.is_array) for n in nets))
        self.subnet_arrays = jax.tree_map(lambda *a: jnp.stack(a), *arrays)
        self.subnet_static = statics[0]

        # 4) 归一化
        self._centres = (self.xmins_all + self.xmaxs_all) / 2
        self._scales = (self.xmaxs_all - self.xmins_all) / 2

    # ----------------- 前向 -----------------
    @eqx.filter_jit
    def raw_output(self, x):
        if x.ndim == 1:
            x = x[None, :]

        def eval_one(subnet_p, centre, scale, mu_min, sd_min, mu_max, sd_max):
            net = eqx.combine(subnet_p, self.subnet_static)

            w_min = jax.nn.sigmoid((x[:, 0] - mu_min[0]) / sd_min[0])
            w_max = jax.nn.sigmoid((mu_max[0] - x[:, 0]) / sd_max[0])
            wm = w_min * w_max

            xi = (x - centre) / jnp.maximum(scale, self.eps)   # (batch, 2)
            xi = xi.T                                          # (2, batch)
            raw = net(xi).squeeze()  
            return wm * raw

        all_out = jax.vmap(eval_one, in_axes=(0, 0, 0, 0, 0, 0, 0))(
            self.subnet_arrays, self._centres, self._scales,
            self.w_mu_min, self.w_sd_min, self.w_mu_max, self.w_sd_max
        )
        return jnp.sum(all_out, axis=0)[..., None]             # (batch,1)

    @eqx.filter_jit
    def __call__(self, x):
        return self.ansatz(x, self.raw_output(x))


# ==============================================================
# 3. 训练器
# ==============================================================
def _chunkify(xy, chunk):
    pad = (-len(xy)) % chunk
    if pad:
        xy = jnp.pad(xy, ((0, pad), (0, 0)))
    n = len(xy) // chunk
    return xy.reshape(n, chunk, -1), n


def train(key, model, problem, colloc, lr=3e-4, steps=100, chunk=1024):
    params, static = eqx.partition(model, eqx.is_array)
    build = lambda p: eqx.combine(p, static)

    def loss(p, xy):
        return problem.residual(build(p), xy)

    vg = jax.value_and_grad(loss)
    opt = optax.adam(lr)
    state = opt.init(params)

    @eqx.filter_jit
    def step(p, s, xy):
        xy_ch, n = _chunkify(xy, chunk)
        zero = jax.tree_map(jnp.zeros_like, p)

        def body(carry, xy_c):
            g_acc, l_acc = carry
            l_i, g_i = vg(p, xy_c)
            g_acc = jax.tree_map(lambda a, g: a + g / n, g_acc, g_i)
            l_acc = l_acc + l_i / n
            return (g_acc, l_acc), None

        (g_tot, l_tot), _ = jax.lax.scan(body, (zero, 0.), xy_ch)
        upd, s = opt.update(g_tot, s, p)
        p = eqx.apply_updates(p, upd)
        return p, s, l_tot

    print("JIT compiling ...", end="", flush=True)
    step(params, state, colloc)
    print(" done")

    for _ in trange(steps, desc="Training"):
        params, state, loss_val = step(params, state, colloc)

    print(f"Final loss: {loss_val.item():.4e}")
    return build(params)


# ==============================================================
# 4. 主程序
# ==============================================================
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    key, k_model, k_pts = jax.random.split(key, 3)

    problem = FirstOrderFreq68()
    mlp_cfg = dict(in_size=2, out_size=1, width_size=64, depth=3)
    subdomains = [(jnp.array([0., 0.]), jnp.array([1., 1.]))]

    print("Initializing model...")
    model = FBPINN(key=k_model,
                   subdomains=subdomains,
                   mlp_cfg=mlp_cfg,
                   ansatz=problem.ansatz)

    print("Generating collocation points...")
    colloc = jax.random.uniform(k_pts, (10_000, 2))

    trained = train(key, model, problem, colloc,
                    steps=100, chunk=1024)

    print("\n✅ Script completed successfully!")
