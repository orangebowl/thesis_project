import numpy as np
import matplotlib.pyplot as plt

# 定义 ansatz 函数
def ansatz_A(x):
    return (1 - np.exp(-x)) * (1 - np.exp(-(8 - x)))

# 在 [0,8] 范围内取 200 个点
x_vals = np.linspace(0, 8, 200)
A_vals = ansatz_A(x_vals)

# 绘图
plt.figure(figsize=(6,4))
plt.plot(x_vals, A_vals, label=r"$B(x) = (1 - e^{-x})(1 - e^{-(8 - x)})$")
plt.xlabel("x")
plt.ylabel("A(x)")
plt.title("Ansatz Boundary Factor")
plt.grid(True)
plt.legend()
plt.show()
