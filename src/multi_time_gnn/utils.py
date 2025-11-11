from pathlib import Path
from typing import Literal
import logging
from datetime import datetime

import torch
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


def get_logger(name: str = "main", level: int = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)

    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def register_model(model: torch.nn.Module, config: Box):
    dir_path = Path(config.output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    model_path = dir_path / (model.__class__.__name__ + ".pt")
    torch.save(model.state_dict(), model_path)
    cfg_path = dir_path / "config.yaml"
    with open(cfg_path, "w") as file:
        yaml.dump(config.to_dict(), file)


def load_model(
    model_class: torch.nn.Module, model_dir: Path, config: Box
) -> torch.nn.Module:
    model_path = model_dir / (model_class.__name__ + ".pt")
    model = model_class(config)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def get_latest_dir(base_path: Path) -> Path:
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {base_path}")

    latest_subdir = max(subdirs, key=lambda d: d.stat().st_mtime)
    return latest_subdir
