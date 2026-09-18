import os
from glob import glob

from safetensors import safe_open
from torch import nn


def load_model(model: nn.Module, path: str) -> None:
    files = sorted(glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found under {path}")

    params = dict(model.named_parameters())
    packed_items = tuple(model.packed_modules_mapping.items())

    for file in files:
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                tensor = f.get_tensor(weight_name)
                for k, (v, shard_id) in packed_items:
                    if k in weight_name:
                        param = params[weight_name.replace(k, v)]
                        param.weight_loader(param, tensor, shard_id)
                        break
                else:
                    param = params[weight_name]
                    param.weight_loader(param, tensor)
