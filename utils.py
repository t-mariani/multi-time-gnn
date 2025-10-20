from typing import Literal
import yaml
from box import Box


def load_config(config_path="config.yaml", return_type: Literal["dict", "box"] = "box"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    if return_type == "dict":
        return config
    elif return_type == "box":
        return Box(config)
    else:
        raise ValueError("return_type not supported, use either 'dict' or 'box'")
