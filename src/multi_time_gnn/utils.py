from pathlib import Path
import random
from typing import Literal
import logging
from datetime import datetime

import numpy as np
import torch
import yaml
from box import Box
from torch.utils.tensorboard import SummaryWriter


def load_config(config_path="config.yaml", return_type: Literal["dict", "box"] = "box"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    if return_type == "dict":
        return config
    elif return_type == "box":
        return Box(config)
    else:
        raise ValueError("return_type not supported, use either 'dict' or 'box'")

def keep_config_model_kind(config):
    """
    if the model kind is AR local, it throws away the MTGNN config
    if the model kind is MTGNN, it throws away the AR local config
    """
    if config.model_kind == "AR_local":
        config.update(config.AR_local)
    elif config.model_kind == "AR_global":
        
    else:
        config.update(config.MTGNN)
    config.pop('MTGNN', None)
    config.pop('AR_global', None)
    return config


def get_logger(name: str = "main", level: int = None, path_file:str=None) -> logging.Logger:
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)

    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if path_file is not None:
            fh = logging.FileHandler(path_file)
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


def register_model(model: torch.nn.Module, config: Box ):
    dir_path = Path(config.output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    if config.model_kind == "MTGNN":
        model_path = dir_path / (model.__class__.__name__ + ".pt")
        torch.save(model.state_dict(), model_path)
    elif config.model_kind == "AR_local":
        np.save(f'{dir_path}/model.npy', model.best_lags)
    
    # Save config for reproducibility
    cfg_path = dir_path / "config.yaml"
    with open(cfg_path, "w") as file:
        yaml.dump(config.to_dict(), file)


def load_model(
    model_class, model_dir: Path, config: Box
):
    if config.model_kind == "MTGNN":
        model = model_class(config)
        model_path = model_dir / (model_class.__name__ + ".pt")
        model.load_state_dict(torch.load(model_path))
        model.eval()
    elif config.model_kind == "AR_local":
        model = model_class(config)
        model_path = model_dir / ("model.npy")
        model.best_lags = np.load(model_path)
    return model


def get_latest_dir(base_path: Path) -> Path:
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {base_path}")

    latest_subdir = max(subdirs, key=lambda d: d.stat().st_mtime)
    return latest_subdir


def set_all_global_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_tensorboard_writer(config):
    """Creates a TensorBoard SummaryWriter in the output directory"""
    # Saves logs to: saved_models/YYYYMMDD-HHMMSS/runs/
    log_dir = Path(config.output_dir) / config.tensorboard_dir
    return SummaryWriter(log_dir=log_dir)