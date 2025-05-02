import yaml
import jax

def get_activation(name):
    mapping = {
        "tanh": jax.nn.tanh,
        "relu": jax.nn.relu,
        "sigmoid": jax.nn.sigmoid,
        "gelu": jax.nn.gelu,
    }
    return mapping.get(name, jax.nn.tanh)  # 默认 tanh

def load_mlp_config(path="config/fbpinn.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    mlp_cfg = cfg["mlp"]
    mlp_cfg["activation"] = get_activation(mlp_cfg["activation"])
    return mlp_cfg