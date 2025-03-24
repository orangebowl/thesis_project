import numpy as np
import matplotlib.pyplot as plt

# 定义区间
x_min, x_max = 0.0, 8.0

# 使用更大的系数，令相位函数为 phi(x) = (pi/4)*x^2
def phi(x):
    return (np.pi / 4.0) * x**2

def u_exact(x):
    return np.sin(phi(x))

# 定义右端项 f(x) = -u''(x)
def f_pde(x):
    # u'(x) = cos(phi(x)) * phi'(x) = cos(phi(x)) * (pi/2*x)
    # u''(x) = d/dx[cos(phi(x))*(pi/2*x)]
    #       = (pi/2)*cos(phi(x)) - (pi^2/4)*x^2*sin(phi(x))
    return (np.pi**2 / 4.0) * x**2 * np.sin(phi(x)) - (np.pi / 2.0) * np.cos(phi(x))

# 生成数据用于绘图
x_plot = np.linspace(x_min, x_max, 400)
u_val = u_exact(x_plot)
f_val = f_pde(x_plot)

# 绘制 exact solution u(x)
plt.figure(figsize=(8,4))
plt.plot(x_plot, u_val, 'b-', label=r'$u_{\rm exact}(x)=\sin\Bigl(\frac{\pi x^2}{4}\Bigr)$')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Exact solution')
plt.legend()
plt.grid(True)
plt.show()

# 绘制右端项 f(x)
plt.figure(figsize=(8,4))
plt.plot(x_plot, f_val, 'r-', label=r'$f(x)$')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Source term for the PDE')
plt.legend()
plt.grid(True)
plt.show()

# 输出边界值检查
print("u_exact(0) =", u_exact(0))
print("u_exact(8) =", u_exact(8))
