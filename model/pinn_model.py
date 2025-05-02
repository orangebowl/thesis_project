import jax
import equinox as eqx

class PINN(eqx.Module):
    mlp: eqx.nn.MLP
    ansatz: callable = eqx.static_field()

    def __init__(self, key, ansatz, width=128, depth=5):
        self.mlp = eqx.nn.MLP(
            in_size=1, out_size=1,
            width_size=width, depth=depth,
            activation=jax.nn.tanh, key=key
        )
        self.ansatz = ansatz

    # ---------- 新增：同时支持批量 & 单点 ----------
    def __call__(self, x):
        """
        x : (...,)            – 1-D 或更高维都行  
        返回 : (...,)
        """
        x_in  = x[..., None]                 # (...,1)
        nn_out = self.mlp(x_in)[..., 0]      # (...,)
        return self.ansatz(x, nn_out)

    # ---------- 新增：配合 FBPINN 接口 ----------
    def total_solution(self, x):
        return self(x)
