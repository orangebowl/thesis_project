# model/pinn_model.py

import jax
import jax.numpy as jnp
import equinox as eqx
from model.Networks import FCN

class PINN(eqx.Module):
    """
    用自定义的 FCN 来替代原来的 equinox.nn.MLP。
    Args:
        key      : JAX PRNGKey，用于初始化 FCN 的参数
        ansatz   : 边界条件 ansatz 函数，签名 (x, nn_out) -> u(x)
        mlp_conf : dict，包含以下字段：
            - in_size    : 输入维度（1D 时传 1，2D 时传 2）
            - out_size   : 输出维度（一般 PINN 输出 u(x) 标量，传 1 即可）
            - width_size : 隐藏层宽度
            - depth      : 隐藏层数目（不含输出层）
            - activation : 隐藏层激活函数，例如 jax.nn.tanh
    """

    net: FCN
    ansatz: callable = eqx.static_field()
    in_dim: int   = eqx.static_field()
    out_dim: int  = eqx.static_field()

    def __init__(
        self,
        *,
        key: jax.Array,
        ansatz: callable,
        mlp_config: dict,
    ):
        # 从 mlp_conf 解出参数
        in_size    = mlp_config["in_size"]
        out_size   = mlp_config["out_size"]
        width_size = mlp_config["width_size"]
        depth      = mlp_config["depth"]
        activation = mlp_config["activation"]

        self.in_dim  = in_size
        self.out_dim = out_size
        self.ansatz  = ansatz

        # FCN 的 layer_sizes: [in_size, width_size, ..., width_size, out_size]
        hidden_sizes = [width_size] * depth
        layer_sizes  = [in_size, *hidden_sizes, out_size]

        # 构造 FCN：
        #   FCN 需要的参数： layer_sizes, key, activation
        self.net = FCN(
            layer_sizes = layer_sizes,
            key         = key,
            activation  = activation,
        )

    def __call__(self, x):
        """
        前向计算：
          - x 可能的形状：
              * 1D 情况: (N,)  → N 个一维样本
              * 2D 情况: (N, 2) → N 个二维样本
              * 单点 2D: (2,) → 1 个样本
          返回：
            ansatz(x, nn_out)，输出形状与 ansatz 定义有关，通常是 (N,1) 或 (N,)
        """
        x = jnp.atleast_1d(x)

        if x.ndim == 1:
            # 如果 in_dim == 1，那么 x 形如 (N,) 代表 N 个一维样本
            if self.in_dim == 1:
                x_in = x[:, None]   # 变成 (N,1)
            else:
                # in_dim > 1，则认为 x 是单点 (2,) 或更高维度
                x_in = x[None, :]   # 变成 (1, in_dim)
        elif x.ndim == 2:
            # 已经是 (N, in_dim)
            x_in = x
        else:
            raise ValueError(f"PINN 只支持 in_dim={self.in_dim} 的情形，但收到 ndim={x.ndim}")

        # 用 FCN 做前向预测：输出形状 (batch, out_size)
        nn_out = self.net(x_in)  # shape = (batch, out_size)
        # 如果 out_size==1，就 squeeze 掉最后一维
        if self.out_dim == 1:
            nn_out = nn_out[:, 0]  # shape = (batch,)

        # 把原始的 x（可能是 (N,) 或 (2,)）连同 nn_out 一起传给 ansatz
        return self.ansatz(x, nn_out)
