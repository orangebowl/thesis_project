import jax
import jax.numpy as jnp
import equinox as eqx


class PINN(eqx.Module):
    mlp: eqx.nn.MLP
    ansatz: callable = eqx.static_field()  # 声明为静态字段，不参与梯度计算

    def __init__(self, key, ansatz):
        self.mlp = eqx.nn.MLP(
            in_size=1, out_size=1, width_size=128, depth=5,
            activation=jax.nn.tanh, key=key
        )
        self.ansatz = ansatz  # 将 ansatz 作为参数存储

    def __call__(self, x):
        nn_out = self.mlp(x[jnp.newaxis])[0]
        return self.ansatz(x, nn_out)  # 使用传入的 ansatz 函数