"""
main.py

完整可运行示例：将 ResPOUNet2D（Two‐phase LSGD + ResNet‐generated PoU）直接内嵌到脚本中，
然后通过子类化将它注入到 Equinox FBPINN 中，完成从 PoU 训练 → FBPINN 训练的全流程。
"""

from __future__ import annotations
import math, pathlib, itertools
import jax, jax.numpy as jnp, optax
from jax import random, jacfwd, jacrev, vmap
import equinox as eqx
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1) ResPOUNet2D + Two‐phase LSGD
# -----------------------------------------------------------------------------
def box_init(rng, w_in, w_out, m, delta, L, l):
    p_key, n_key = random.split(rng)
    p = m * random.uniform(p_key, (w_in, w_out))
    n_hat = random.normal(n_key, (w_in, w_out))
    n_hat /= jnp.linalg.norm(n_hat, axis=0, keepdims=True) + 1e-9
    p_max = m * jnp.maximum(0.0, jnp.sign(n_hat))
    k = 1.0 / ((L - 1) * (jnp.sum((p_max - p) * n_hat, axis=0) + 1e-9))
    return k * n_hat, jnp.sum(k * n_hat * p, axis=0)

def init_linear_layer(rng, w_in, w_out, layer_idx, L):
    l = layer_idx + 1
    m = (1.0 + 1.0/(L-1))**l
    W, b = box_init(rng, w_in, w_out, m, 1.0/L, L, l)
    return {"W": W, "b": b}

def apply_linear(p, x):
    return x @ p["W"] + p["b"]

class ResPOUNet2D:
    def __init__(self, input_dim=2, num_partitions=4,
                 hidden_dim=32, depth=6, key=None):
        assert depth >= 3
        if key is None: key = random.PRNGKey(0)
        keys = random.split(key, depth)
        layers = []
        layers.append(init_linear_layer(keys[0], input_dim, hidden_dim, 0, depth))
        for i in range(1, depth-1):
            layers.append(init_linear_layer(keys[i], hidden_dim, hidden_dim, i, depth))
        layers.append(init_linear_layer(keys[-1], hidden_dim, num_partitions, depth-1, depth))
        self._init = layers
        self.C = num_partitions

    def init_params(self):
        return jax.tree_map(lambda z: z.copy(), self._init)

    def forward(self, params, x):
        h = jax.nn.relu(apply_linear(params[0], x))
        for i in range(1, len(params)-1):
            h = h + jax.nn.relu(apply_linear(params[i], h))
        logits = apply_linear(params[-1], h)
        return jax.nn.softmax(logits, axis=-1)

def toy_func(xy):
    x,y = xy[...,0], xy[...,1]
    return jnp.sin(2*jnp.pi*x)*jnp.sin(2*jnp.pi*y)

def _design(xy):
    x,y = xy[:,0], xy[:,1]
    return jnp.stack([jnp.ones_like(x), x, y, x**2, x*y, y**2], axis=-1)

def fit_local_poly2(xy, f, w, lam=0.0):
    A, y = _design(xy), f[:,None]
    def solve(weights):
        Aw = A * weights[:,None]
        M  = A.T @ Aw
        b  = (Aw.T @ y).squeeze(-1)
        return jnp.linalg.solve(M + lam*jnp.eye(6), b)
    return vmap(solve, in_axes=1, out_axes=0)(w)

def predict(xy, coeffs, part):
    return jnp.sum(part * (_design(xy) @ coeffs.T), axis=1)

def _run_lsgd(model, params, xy, f,
              n_epochs, lr, lam0, rho, n_stag):
    lam = jnp.array(lam0); best, stag = jnp.inf, 0
    @jax.jit
    def loss_fn(p, lam_):
        part   = model.forward(p, xy)
        coeffs = fit_local_poly2(xy, f, part, lam_)
        pred   = predict(xy, coeffs, part)
        return jnp.mean((pred - f)**2)
    valgrad = jax.jit(lambda p, lam_: jax.value_and_grad(lambda pp: loss_fn(pp, lam_))(p))
    opt = optax.adam(lr); state = opt.init(params)
    ep = 0
    for _ in range(n_epochs):
        loss, grads = valgrad(params, lam)
        updates, state = opt.update(grads, state)
        params = optax.apply_updates(params, updates)
        if loss < best - 1e-12:
            best, stag = loss, 0
        else:
            stag += 1
        if stag > n_stag:
            lam *= rho; stag = 0
        ep += 1
    return params, ep

def train_two_phase(pou_net, xy, f,
                    n_pre=4000, n_post=2000,
                    lr_pre=1e-3, lr_post=5e-4,
                    lam0=1e-3, rho=0.99, n_stag=100):
    p = pou_net.init_params()
    p, ep = _run_lsgd(pou_net, p, xy, f, n_pre,  lr_pre, lam0, rho, n_stag)
    p, _  = _run_lsgd(pou_net, p, xy, f, n_post, lr_post, 0.0, 1.0, n_stag)
    return p

# -----------------------------------------------------------------------------
# 2) 子类化 Equinox FBPINN，把 PoU 注入 window
# -----------------------------------------------------------------------------
def glorot_init(rng,in_,out_):
    lim = jnp.sqrt(6/(in_+out_))
    return random.uniform(rng,(in_,out_),minval=-lim,maxval=lim)

def init_fcn(rng, sizes):
    keys = random.split(rng, len(sizes)-1)
    params = []
    for k,(m,n) in zip(keys, zip(sizes[:-1], sizes[1:])):
        params.append({"W": glorot_init(k,m,n), "b": jnp.zeros((n,))})
    return params

def fcn_forward(params, x):
    h = x
    for lyr in params[:-1]:
        h = jnp.tanh(h @ lyr["W"] + lyr["b"])
    return (h @ params[-1]["W"] + params[-1]["b"]).squeeze(-1)

# 原始 FBPINN （保持不动）
class FBPINN(eqx.Module):
    subnets: tuple
    ansatz: callable = eqx.static_field()
    subdomains: tuple = eqx.static_field()
    num_subdomains: int = eqx.static_field()

    def __init__(self, key, num_subdomains, ansatz, subdomains, mlp_config):
        self.ansatz = ansatz
        self.subdomains = subdomains
        self.num_subdomains = num_subdomains
        keys = random.split(key, num_subdomains)
        self.subnets = tuple(
            eqx.nn.MLP(
                in_size=mlp_config["in_size"],
                out_size=mlp_config["out_size"],
                width_size=mlp_config["width_size"],
                depth=mlp_config["depth"],
                activation=mlp_config["activation"],
                key=k,
            )
            for k in keys
        )

    def subdomain_window(self, i, x, tol=1e-8):
        # 旧版硬阈值：检查几何……
        left, right = self.subdomains[i]
        inside = jnp.all((x >= left-tol) & (x <= right+tol), axis=-1)
        return inside.astype(x.dtype)

    def subdomain_pred(self, i, x):
        x = jnp.atleast_2d(x)
        # normalize, vmap(subnet)……
        return fcn_forward((), x)  # stub

    def __call__(self, x):
        # stub
        return jnp.zeros((x.shape[0],))

# 子类化：注入 PoU
class FBPINNWithWindow(FBPINN):
    pou_net:    ResPOUNet2D = eqx.static_field()
    pou_params: any         = eqx.static_field()

    def __init__(self, key, num_subdomains, ansatz, subdomains, mlp_config,
                 pou_net, pou_params):
        super().__init__(key, num_subdomains, ansatz, subdomains, mlp_config)
        object.__setattr__(self, "pou_net",    pou_net)
        object.__setattr__(self, "pou_params", pou_params)

    def subdomain_window(self, i, x, tol=1e-8):
        x = jnp.atleast_2d(x)
        w_all = self.pou_net.forward(self.pou_params, x)
        return w_all[:, i]

    def __call__(self, x):
        """Override total_solution: sum_j ω_j(x) * subnet_j(x) then ansatz"""
        x = jnp.atleast_2d(x)
        total = 0.0
        for j in range(self.num_subdomains):
            wj = self.subdomain_window(j, x)
            uj = jax.vmap(self.subnets[j])( (x - 0.5)*(2.0) )[:,0]  # normalize stub
            total += wj * uj
        return self.ansatz(x, total)

# -----------------------------------------------------------------------------
# 3) PDE 定义 & 训练循环
# -----------------------------------------------------------------------------
def exact_u(xy):
    x,y = xy[...,0], xy[...,1]
    return jnp.sin(2*jnp.pi*x**2)*jnp.sin(2*jnp.pi*y**2)

def rhs_f(xy):
    x,y = xy[...,0], xy[...,1]
    sinx, cosx = jnp.sin(2*jnp.pi*x**2), jnp.cos(2*jnp.pi*x**2)
    siny, cosy = jnp.sin(2*jnp.pi*y**2), jnp.cos(2*jnp.pi*y**2)
    return -4*jnp.pi*(siny*cosx + sinx*cosy) + 16*jnp.pi**2*(x**2+y**2)*sinx*siny

def pde_residual(params, model, xy):
    def u_fn(pt):
        return model(pt.reshape(1,2)).squeeze()
    hess = jacfwd(jacrev(u_fn))
    H = vmap(hess)(xy)
    lap = jnp.trace(H, -2, -1)
    f = rhs_f(xy)
    return jnp.mean((-lap - f)**2)

def train_fbpinn(model, params, colloc_pts, steps=2000, lr=1e-3):
    opt = optax.adam(lr); state = opt.init(params)
    loss_hist = []
    for i in range(steps):
        def loss_fn(p):
            return sum(pde_residual(p, model, xy) for xy in colloc_pts)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, state = opt.update(grads, state)
        params = optax.apply_updates(params, updates)
        loss_hist.append(loss)
        if i%500==0:
            print(f"step {i} | loss {loss:.3e}")
    return params, loss_hist

# -----------------------------------------------------------------------------
# 4) 主流程
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 4.1 训练 PoU
    key = random.PRNGKey(0)
    pou_net = ResPOUNet2D(input_dim=2, num_partitions=4, hidden_dim=32, depth=6, key=key)
    xs = jnp.linspace(0,1,40)
    grid = jnp.stack(jnp.meshgrid(xs,xs),axis=-1).reshape(-1,2)
    fgrid = toy_func(grid)
    pou_params = train_two_phase(pou_net, grid, fgrid,
                                n_pre=3000, n_post=1000,
                                lr_pre=1e-3, lr_post=5e-4,
                                lam0=1e-3, rho=0.99, n_stag=50)

    # 4.2 生成子域 & collocation points
    domain = (jnp.array([0.,0.]), jnp.array([1.,1.]))
    n_sub_per_dim = 2; overlap=0.3
    # 简单 grid 子域
    xs = jnp.linspace(0,1,n_sub_per_dim+1)
    subs = []
    for i in range(n_sub_per_dim):
        for j in range(n_sub_per_dim):
            left = jnp.array([xs[i],   xs[j]])
            right= jnp.array([xs[i+1], xs[j+1]])
            subs.append((left, right))
    C = pou_net.C
    Ncolloc=1000
    Xc = random.uniform(random.PRNGKey(1),(Ncolloc,2))
    Wc = pou_net.forward(pou_params, Xc)
    colloc = [ Xc[Wc[:,k]>1e-6] for k in range(C) ]

    # 4.3 构造并训练 FBPINNWithWindow
    mlp_cfg={"in_size":2,"out_size":1,"width_size":8,"depth":2,"activation":jax.nn.tanh}
    def ansatz_fn(x,u): return u
    model = FBPINNWithWindow(random.PRNGKey(2), C, ansatz_fn, subs, mlp_cfg,
                             pou_net, pou_params)
    params0 = eqx.filter(model, eqx.is_inexact_array)
    params_trained, loss_hist = train_fbpinn(model, params0, colloc, steps=2000, lr=1e-3)

    # 4.4 可视化
    xs_t = jnp.linspace(0,1,50)
    Xtest = jnp.stack(jnp.meshgrid(xs_t,xs_t),axis=-1).reshape(-1,2)
    Up = model(Xtest).reshape(50,50)
    Ue = exact_u(Xtest).reshape(50,50)

    fig,ax=plt.subplots(1,2,figsize=(8,4))
    ax[0].imshow(Up,origin="lower",extent=[0,1,0,1]); ax[0].set_title("FBPINN+PoU")
    ax[1].imshow(jnp.abs(Up-Ue),origin="lower",extent=[0,1,0,1]); ax[1].set_title("Error")
    plt.show()
