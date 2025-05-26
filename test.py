import os
import sys
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import equinox as eqx
import traceback # 用于打印更详细的错误信息
from functools import partial # 导入 partial

# --- 辅助函数：窗函数 ---
def sigmoid_window_single_point(xmin, xmax, wmin, wmax, x, tol=1e-8):
    """
    Window function for a SINGLE point with shape (xdim).
    xmin, xmax, wmin, wmax are also expected to be of shape (xdim,).
    Returns a window weight of shape (1,).
    """
    t = jnp.log((1 - tol) / tol)
    
    # Add a small epsilon to sd to prevent division by zero if wmin/wmax were theoretically zero,
    # though fixed_transition_width should prevent this.
    epsilon = 1e-9 # 避免除以零
    
    sd_min = wmin / (2 * t + epsilon) # 确保分母不为零
    sd_max = wmax / (2 * t + epsilon) # 确保分母不为零

    mu_min = xmin + wmin / 2
    mu_max = xmax - wmax / 2
    
    # 确保 sd_min 和 sd_max 不是太小以至于导致溢出或 NaN
    # (x - mu_min) / (sd_min + epsilon)
    # (mu_max - x) / (sd_max + epsilon)

    ws = jax.nn.sigmoid((x - mu_min) / (sd_min + epsilon)) * \
         jax.nn.sigmoid((mu_max - x) / (sd_max + epsilon))

    w = jnp.prod(ws, axis=0, keepdims=True) # Output shape: (1,)
    return w

def vectorized_sigmoid_window_func(xmins_all_subdomains, xmaxs_all_subdomains,
                                   wmins_all_subdomains, wmaxs_all_subdomains,
                                   x_batch_NxD, tol=1e-8):
    """
    Calculates window weights for a batch of points across all subdomains
    using the sigmoid_window_single_point function.
    """
    def calculate_weights_for_one_point(x_one_point_xdim,
                                        xmins_sXd, xmaxs_sXd,
                                        wmins_sXd, wmaxs_sXd):
        weights_per_subdomain_for_one_point = jax.vmap(
            sigmoid_window_single_point,
            in_axes=(0, 0, 0, 0, None, None),
            out_axes=0
        )(xmins_sXd, xmaxs_sXd, wmins_sXd, wmaxs_sXd, x_one_point_xdim, tol)
        return weights_per_subdomain_for_one_point.squeeze(axis=-1)

    all_weights_NxS = jax.vmap(
        calculate_weights_for_one_point,
        in_axes=(0, None, None, None, None),
        out_axes=0
    )(x_batch_NxD, xmins_all_subdomains, xmaxs_all_subdomains, wmins_all_subdomains, wmaxs_all_subdomains)
    return all_weights_NxS

# --- 辅助函数：数据生成 (简化版占位符) ---
def generate_subdomains(domain_tuple, n_sub, overlap):
    """简化版的子域生成函数"""
    min_val = domain_tuple[0][0]
    max_val = domain_tuple[1][0]
    
    if n_sub == 0:
        return []
    if n_sub == 1:
        return [(domain_tuple[0], domain_tuple[1])]

    domain_length = max_val - min_val
    subdomain_length = domain_length / (n_sub - (n_sub - 1) * overlap)
    step = subdomain_length * (1 - overlap)
    
    subdomains = []
    current_min = min_val
    for i in range(n_sub):
        current_max = current_min + subdomain_length
        # 确保最后一个子域的右边界不超过全局域的右边界
        if i == n_sub - 1:
            current_max = max_val
            current_min = max_val - subdomain_length # 可能需要调整以保持长度
            if current_min < domain_tuple[0][0] and n_sub > 1 : # 如果调整后左边界超出，则需要重新计算
                 # 这是一个简化版本，实际情况可能更复杂
                 s_min_val = domain_tuple[0][0] + i * step
                 s_max_val = s_min_val + subdomain_length
                 if s_max_val > max_val: s_max_val = max_val
                 if i == n_sub -1 : s_max_val = max_val #确保最后一个子域的右边界是全局右边界
                 s_min_val = s_max_val - subdomain_length
                 current_min = s_min_val
                 current_max = s_max_val


        subdomains.append((jnp.array([current_min]), jnp.array([current_max])))
        current_min += step
        if current_min > max_val and i < n_sub -1: # 防止超出太多
            break
            
    # 修正最后一个子域确保它到达 domain_max
    if subdomains and subdomains[-1][1][0] < max_val:
        # For simplicity, just extend the last one or adjust.
        # This stub is very basic.
        # A more robust version would ensure coverage and overlap precisely.
        last_min, _ = subdomains[-1]
        if n_sub > 1 : # 只有多个子域时才可能需要调整最后一个的左边界
            prev_max = subdomains[-2][1][0] if n_sub > 1 else domain_tuple[0][0]
            # 简单的重叠逻辑
            desired_last_min = max_val - subdomain_length
            # 如果最后一个子域的左边界因为之前的子域延伸而变得太小，
            # 并且与前一个子域的重叠部分比预期的要大，这部分逻辑需要小心处理。
            # 这里采用简化处理：
            adjusted_min = jnp.maximum(last_min[0], max_val - subdomain_length)
            # 确保 adjusted_min 至少是前一个子域结束减去重叠长度
            if n_sub > 1:
                 overlap_abs = subdomain_length * overlap
                 min_from_prev = subdomains[-2][1][0] - overlap_abs
                 adjusted_min = jnp.maximum(adjusted_min, min_from_prev)


            subdomains[-1] = (jnp.array([adjusted_min]), domain_tuple[1])
        else: # n_sub == 1
            subdomains[-1] = (domain_tuple[0], domain_tuple[1])


    # 确保所有子域都在全局域内
    final_subdomains = []
    for s_min, s_max in subdomains:
        s_min_clipped = jnp.maximum(s_min, domain_tuple[0])
        s_max_clipped = jnp.minimum(s_max, domain_tuple[1])
        if s_min_clipped < s_max_clipped: # 确保子域有效
             final_subdomains.append((s_min_clipped, s_max_clipped))
    
    if not final_subdomains and n_sub > 0: # 如果没有任何有效子域，但要求有子域
        return [(domain_tuple[0], domain_tuple[1])] # 返回整个域作为单个子域

    return final_subdomains


def generate_collocation_points(domain, subdomains_list, n_points_per_subdomain, seed):
    """简化版的配点生成函数"""
    key = jax.random.PRNGKey(seed)
    all_points = []
    
    if not subdomains_list: # 如果没有子域，则在整个域上生成点
        if domain[0] is not None and domain[1] is not None:
            min_val, max_val = domain[0][0], domain[1][0]
            key, subkey = jax.random.split(key)
            # Assuming 1D domain for simplicity in linspace
            points = jnp.linspace(min_val, max_val, n_points_per_subdomain).reshape(-1, 1)
            all_points.append(points)
        return all_points, [] # 返回空的边界点列表


    for i, (s_min_arr, s_max_arr) in enumerate(subdomains_list):
        s_min, s_max = s_min_arr[0], s_max_arr[0]
        key, subkey = jax.random.split(key)
        # JAX 的 linspace 需要具体数值，而不是数组
        points = jnp.linspace(s_min.item(), s_max.item(), n_points_per_subdomain)
        # points = jax.random.uniform(subkey, shape=(n_points_per_subdomain,), minval=s_min, maxval=s_max)
        all_points.append(points.reshape(-1, 1)) # 确保是 (N, 1)
        
    # 此简化版本不生成边界点，所以返回空列表
    boundary_points_global = [] 
    return all_points, boundary_points_global


# --- PDE 定义 ---
class PDEProblem:
    domain = (None, None)
    def residual(self, model, x): pass
    def exact(self, x): pass
    def ansatz(self, x, nn_out): pass

class CosineODE(PDEProblem):
    omega = 15.0
    domain = (-2 * jnp.pi, 2 * jnp.pi)

    @staticmethod
    def ansatz(x, nn_out):
        if x.shape[-1] == 1:
            x_squeezed = x[:, 0]
        else:
            x_squeezed = x 
        return jnp.tanh(CosineODE.omega * x_squeezed) * nn_out

    @staticmethod
    def exact(x):
        if x.shape[-1] == 1:
            x_squeezed = x[:, 0]
        else:
            x_squeezed = x
        return jnp.sin(CosineODE.omega * x_squeezed) / CosineODE.omega

    def _single_res(self, model, x_batch):
        u_x = jax.vmap(jax.grad(lambda y_vec: model(y_vec)[0]))(x_batch)
        return jnp.mean((u_x - jnp.cos(self.omega * x_batch))**2)

    def residual(self, model, x):
        if isinstance(x, (list, tuple)):
            losses = [self._single_res(model, xi) for xi in x]
            return jnp.sum(jnp.stack(losses))
        else:
            return self._single_res(model, x)

pde_module = CosineODE

# --- FBPINN 类定义 (已修改) ---
class FBPINN(eqx.Module):
    subnets: tuple
    ansatz: callable = eqx.static_field()
    
    xmins_all: jax.Array
    xmaxs_all: jax.Array
    wmins_all_fixed: jax.Array
    wmaxs_all_fixed: jax.Array
    
    num_subdomains: int = eqx.static_field()
    model_out_size: int = eqx.static_field() 
    xdim: int = eqx.static_field()

    domain: tuple = eqx.static_field()

    def __init__(self, key, subdomains_tuple, ansatz, mlp_config, fixed_transition_width):
        self.ansatz = ansatz
        self.model_out_size = mlp_config["out_size"]
        self.xdim = mlp_config["in_size"]
        
        if not subdomains_tuple: # 如果子域列表为空
            # print("Warning: subdomains_tuple is empty. FBPINN will have no subnets.")
            self.num_subdomains = 0
            self.subnets = tuple()
            # 即使没有子域，也需要定义这些属性以避免 Equinox 错误
            # 使用一个占位符形状，或者基于 xdim
            placeholder_shape = (0, self.xdim) if self.xdim > 0 else (0,)
            self.xmins_all = jnp.empty(placeholder_shape, dtype=jnp.float32)
            self.xmaxs_all = jnp.empty(placeholder_shape, dtype=jnp.float32)
            self.wmins_all_fixed = jnp.empty(placeholder_shape, dtype=jnp.float32)
            self.wmaxs_all_fixed = jnp.empty(placeholder_shape, dtype=jnp.float32)
            # 尝试从 mlp_config 或一个默认值设置域
            # 如果 problem_domain_tuple 在外部定义，可能需要一种方式传递它或设为None
            self.domain = (None, None) # 或者一个合理的默认值
        else:
            self.num_subdomains = len(subdomains_tuple)
            # 确保 s[0] 和 s[1] 是 jnp 数组并且具有正确的形状 (xdim,)
            self.xmins_all = jnp.stack([jnp.array(s[0]).reshape(self.xdim) for s in subdomains_tuple])
            self.xmaxs_all = jnp.stack([jnp.array(s[1]).reshape(self.xdim) for s in subdomains_tuple])

            # 如果 xdim=1 且 stack 结果是 (num_subdomains,)，则调整为 (num_subdomains, 1)
            if self.xdim == 1 and self.xmins_all.ndim == 1:
                self.xmins_all = self.xmins_all[:, None]
                self.xmaxs_all = self.xmaxs_all[:, None]
            
            domain_min_arr = jnp.min(self.xmins_all, axis=0)
            domain_max_arr = jnp.max(self.xmaxs_all, axis=0)
            
            if domain_min_arr.ndim == 0: 
                self.domain = (domain_min_arr.item(), domain_max_arr.item())
            elif domain_min_arr.shape[0] == 1: 
                self.domain = (domain_min_arr[0].item(), domain_max_arr[0].item())
            else: 
                self.domain = (domain_min_arr.tolist(), domain_max_arr.tolist())

            self.wmins_all_fixed = jnp.full((self.num_subdomains, self.xdim), fixed_transition_width, dtype=self.xmins_all.dtype)
            self.wmaxs_all_fixed = jnp.full((self.num_subdomains, self.xdim), fixed_transition_width, dtype=self.xmaxs_all.dtype)

            keys = jax.random.split(key, self.num_subdomains)
            self.subnets = tuple(
                eqx.nn.MLP(
                    in_size=self.xdim,
                    out_size=self.model_out_size, 
                    width_size=mlp_config["width_size"],
                    depth=mlp_config["depth"],
                    activation=mlp_config["activation"],
                    key=k
                )
                for k in keys
            )
            
    def _normalize_x_logic(self, i, x_input_nd): # x_input can be (N, xdim) or (xdim,)
        left = self.xmins_all[i]      # Shape (xdim,)
        right = self.xmaxs_all[i]     # Shape (xdim,)
        center = (left + right) / 2.0
        scale = (right - left) / 2.0
        return (x_input_nd - center) / jnp.maximum(scale, 1e-9)


    def total_solution(self, x):
        # x 可以是 (N, xdim) 或 (xdim,)
        # 我们希望处理 (N, xdim) 的批处理输入
        # jnp.atleast_2d 对于 (xdim,) 输入会变成 (1, xdim)
        # 如果 x 是标量, jnp.atleast_2d(jnp.array(x)) -> [[x]]
        if isinstance(x, (float, int)): x = jnp.array([x]) # 转换为数组
        if x.ndim == self.xdim -1 and self.xdim > 0 : # e.g. xdim=1, x.shape=() -> x.shape=(1,)
             x = x.reshape(1, *x.shape) if self.xdim ==1 else x.reshape(*x.shape,1) #TODO: check this logic, better ensure x is (N,xdim) or (xdim)
        
        # 确保 x_2d 是 (N, xdim)
        if x.ndim == self.xdim : # 假设单个点 (xdim,)
            x_2d = x[None, :]
        elif x.ndim == self.xdim +1 and x.shape[1] == self.xdim: # 已经是 (N, xdim)
            x_2d = x
        else: #尝试整形
            try:
                x_2d = x.reshape(-1, self.xdim)
            except:
                 raise ValueError(f"Input x has shape {x.shape}, cannot be reshaped to (-1, {self.xdim})")

        N = x_2d.shape[0]
        
        if N == 0:
            return jnp.empty((0,) if self.model_out_size == 1 else (0, self.model_out_size), dtype=x_2d.dtype)
        
        if self.num_subdomains == 0:
            if self.model_out_size == 1:
                zero_pred = jnp.zeros(N, dtype=x_2d.dtype)
            else:
                zero_pred = jnp.zeros((N, self.model_out_size), dtype=x_2d.dtype)
            return self.ansatz(x_2d, zero_pred)

        all_window_weights = vectorized_sigmoid_window_func(
            self.xmins_all, self.xmaxs_all,
            self.wmins_all_fixed, self.wmaxs_all_fixed,
            x_2d
        )
        
        branch_fns = [
            (lambda current_subnet: lambda operand_x_norm: jax.vmap(current_subnet)(operand_x_norm))(subnet)
            for subnet in self.subnets
        ]

        def loop_body(carry_sum, k_idx): 
            w_k = all_window_weights[:, k_idx]
            x_norm_k = self._normalize_x_logic(k_idx, x_2d)
            raw_out = jax.lax.switch(k_idx, branch_fns, x_norm_k) 
            out_k = raw_out[:, 0] if self.model_out_size == 1 else raw_out
            
            if self.model_out_size > 1:
                updated_sum = carry_sum + w_k[:, None] * out_k
            else: 
                updated_sum = carry_sum + w_k * out_k
            return updated_sum, None

        expected_dtype = x_2d.dtype 
        if self.model_out_size == 1:
            initial_sum = jnp.zeros(N, dtype=expected_dtype) 
        else:
            initial_sum = jnp.zeros((N, self.model_out_size), dtype=expected_dtype)
        
        total_weighted_output, _ = jax.lax.scan(
            loop_body,
            initial_sum,
            jnp.arange(self.num_subdomains)
        )
        return self.ansatz(x_2d, total_weighted_output)

    def __call__(self, x):
        return self.total_solution(x)

# --- 用户的前置代码继续 ---
problem = pde_module()
pde_residual_loss = problem.residual  
u_exact = problem.exact
ansatz = problem.ansatz 
problem_domain_tuple = problem.domain 

# Training hyperparameters
steps = 1000 # 减少步数以便快速测试
lr = 1e-3      
n_sub = 10
overlap = 0.1
n_points_per_subdomain = 20 # 减少点数以便快速测试

mlp_config = {
    "in_size": 1, 
    "out_size": 1, 
    "width_size": 32,
    "depth": 2,
    "activation": jax.nn.tanh,
}

if isinstance(problem_domain_tuple, tuple) and len(problem_domain_tuple) == 2 and \
   isinstance(problem_domain_tuple[0], (int, float)) and \
   isinstance(problem_domain_tuple[1], (int, float)):
    domain_for_generation = (jnp.array([float(problem_domain_tuple[0])]), jnp.array([float(problem_domain_tuple[1])]))
else: # 假设它已经是 (jnp.array, jnp.array) 形式或需要转换
    if isinstance(problem_domain_tuple[0], (int, float)):
        domain_for_generation = (jnp.array([float(problem_domain_tuple[0])]), jnp.array([float(problem_domain_tuple[1])]))
    else: # 已经是 jnp array
        domain_for_generation = problem_domain_tuple


print("Domain used for generating subdomains:", domain_for_generation)
subdomains_list = generate_subdomains(domain_for_generation, n_sub, overlap)
if not subdomains_list and n_sub > 0 : # 确保如果要求子域但生成失败，则至少有一个覆盖整个域的子域
    print(f"Warning: generate_subdomains returned empty list for n_sub={n_sub}. Defaulting to single domain.")
    subdomains_list = [(domain_for_generation[0], domain_for_generation[1])]


print(f"Generated {len(subdomains_list)} subdomains.")
for i, (smin, smax) in enumerate(subdomains_list):
    print(f"  Subdomain {i}: [{smin.item():.4f}, {smax.item():.4f}]")


subdomain_collocation_points, _ = generate_collocation_points(
    domain=domain_for_generation, 
    subdomains_list=subdomains_list,
    n_points_per_subdomain=n_points_per_subdomain, 
    seed=0
)

def print_collocation_ranges(subdomain_points_list):
    if not subdomain_points_list :
        print("No collocation points generated.")
        return
    for i, pts_array in enumerate(subdomain_points_list): 
        pts_array = jnp.asarray(pts_array)
        if pts_array.size == 0:
            print(f"Collocation points for Subdomain {i}: EMPTY")
            continue
        if pts_array.ndim == 1:
            lo = float(pts_array.min())
            hi = float(pts_array.max())
            ranges_str = f"d0: [{lo:.4f}, {hi:.4f}]" 
        else:
            mins = pts_array.min(axis=0)
            maxs = pts_array.max(axis=0)
            ranges_str = ", ".join(
                f"d{j}: [{float(mins[j]):.4f}, {float(maxs[j]):.4f}]"
                for j in range(pts_array.shape[1])
            )
        print(f"Collocation points for Subdomain {i}: {ranges_str} (N={pts_array.shape[0]})")

print_collocation_ranges(subdomain_collocation_points)

# --- 实例化和使用 FBPINN ---
if __name__ == '__main__': 
    key = jax.random.PRNGKey(42) 
    fixed_transition_width = 0.05 
    print(f"\nUsing fixed_transition_width: {fixed_transition_width}")

    model_init_key, _ = jax.random.split(key) 
    
    if not subdomains_list and n_sub > 0: # Safety check if generate_subdomains failed
        print("Error: Subdomain list is empty before FBPINN initialization. Check generate_subdomains.")
        # sys.exit(1) # Exit if critical, or handle as FBPINN init does (0 subdomains)

    fb_pinn_model = FBPINN(
        key=model_init_key,
        subdomains_tuple=subdomains_list, 
        ansatz=ansatz, 
        mlp_config=mlp_config,
        fixed_transition_width=fixed_transition_width
    )
    print(f"FBPINN model initialized with {fb_pinn_model.num_subdomains} subdomains.")
    if fb_pinn_model.domain[0] is not None:
         print(f"Model's effective domain based on subdomains: {fb_pinn_model.domain[0]:.4f} to {fb_pinn_model.domain[1]:.4f}")
    else:
        print("Model's effective domain is undefined (likely 0 subdomains).")


    jitted_model_call = eqx.filter_jit(fb_pinn_model)
    print("Model JIT compiled using eqx.filter_jit.")

    num_test_points = 200
    # 确保 problem_domain_tuple[0] 和 [1] 是数值
    domain_start = float(problem_domain_tuple[0])
    domain_end = float(problem_domain_tuple[1])

    test_points_x = jnp.linspace(
        domain_start, 
        domain_end, 
        num_test_points
    ).reshape(-1, mlp_config["in_size"])

    print(f"Generated {num_test_points} test points from {domain_start} to {domain_end}")

    try:
        print("Attempting a forward pass with the JITted model...")
        predictions = jitted_model_call(test_points_x)
        print("Predictions shape (after ansatz):", predictions.shape) 

        if predictions.ndim == 1:
            plot_predictions = predictions
        elif predictions.ndim > 1 and predictions.shape[-1] == 1 :
            plot_predictions = predictions.flatten()
        else:
            print(f"Warning: Predictions have shape {predictions.shape}, which might not be suitable for 1D plotting directly.")
            plot_predictions = predictions[:, 0].flatten() 

        print("Forward pass successful!")

        if u_exact is not None:
            exact_solution_values = u_exact(test_points_x)
            if exact_solution_values.ndim > 1 and exact_solution_values.shape[-1] == 1:
                plot_exact_solution = exact_solution_values.flatten()
            elif exact_solution_values.ndim == 1:
                plot_exact_solution = exact_solution_values
            else:
                print(f"Warning: Exact solution has shape {exact_solution_values.shape}, might need adjustment for 1D plotting.")
                plot_exact_solution = exact_solution_values[:,0].flatten()

            if plot_predictions.shape != plot_exact_solution.shape:
                print(f"Shape mismatch for error calculation: preds {plot_predictions.shape}, exact {plot_exact_solution.shape}")
            else:
                error = jnp.abs(plot_predictions - plot_exact_solution)
                print(f"Mean absolute error on test points: {jnp.mean(error):.4e}")

            plt.figure(figsize=(10, 6))
            plt.plot(test_points_x.flatten(), plot_predictions, label="FBPINN Prediction")
            plt.plot(test_points_x.flatten(), plot_exact_solution, label="Exact Solution", linestyle='--')
            plt.xlabel("x")
            plt.ylabel("u(x)")
            plt.legend()
            plt.title(f"FBPINN Solution ({fb_pinn_model.num_subdomains} subdomains)")
            plt.grid(True)
            plt.show()

    except Exception as e:
        print("Error during model prediction or plotting:")
        traceback.print_exc()