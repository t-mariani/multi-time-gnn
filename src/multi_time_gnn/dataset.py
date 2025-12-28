from abc import abstractmethod
from typing import Literal
from einops import rearrange
import torch
import numpy as np
from multi_time_gnn.utils import get_logger
from torch.utils.data import Dataset

log = get_logger()

available_datasets = ["electricity", "traffic", "solar", "exchange"]


def read_dataset(
    which: Literal["electricity", "traffic", "solar", "exchange"],
) -> list[list[float]]:
    if which == "electricity":
        path = "../multivariate-time-series-data/electricity/electricity.txt"
    elif which == "traffic":
        path = "../multivariate-time-series-data/traffic/traffic.txt"
    elif which == "solar":
        path = "../multivariate-time-series-data/solar-energy/solar_AL.txt"
    elif which == "exchange":
        path = "../multivariate-time-series-data/exchange_rate/exchange_rate.txt"
    else:
        raise ValueError(
            f"Unknown dataset: {which}, available are {available_datasets}"
        )

    with open(path, "r") as f:
        data = f.readlines()
    data = [list(map(float, line.strip().split(","))) for line in data]

    return np.array(
        data
    ).T  # Return size : NxT 


def split_train_val_test(data, train_ratio=0.7, val_ratio=0.1):
    n_timesteps = data.shape[1]
    train_size = int(n_timesteps * train_ratio)
    val_size = int(n_timesteps * val_ratio)
    train = data[:, :train_size]
    val = data[:, train_size : train_size + val_size]
    test = data[:, train_size + val_size :]
    return train, val, test

class TimeSeriesDataset(Dataset):
    def __init__(self, data, config, length_prediction=1):
        self.data = data
        self.nb_points = config.timepoints_input
        self.device = config.device
        self.length_prediction = length_prediction

    def __len__(self):
        return self.data.shape[-1] - self.nb_points - self.length_prediction

    def __getitem__(self, idx):
        x = self.data[:, idx:idx + self.nb_points]
        y = self.data[:, idx + self.nb_points: idx + self.nb_points + self.length_prediction].squeeze()
        x = x[None, :, :]  # 1xNxT
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


class Normalizer:
    """Abstract Normalizer class"""
    @abstractmethod
    def normalize(self, data):
        pass
    
    @abstractmethod
    def denormalize(self, data):
        pass
    
class ZscoreNormalizer(Normalizer):
    """Z-score Normalizer, normalizes data to have mean 0 and std 1"""
    def __init__(self, data_fit: np.ndarray):
        """data_fit : np.ndarray of shape (N, T) used to compute mean and std"""
        self.mean = data_fit.mean(axis=1)
        self.std = data_fit.std(axis=1)
    
    def normalize(self, data):
        return (data - self.mean[:, None]) / self.std[:, None]
    
    def denormalize(self, data):
        return data * self.std[:, None] + self.mean[:, None]
    
class MinMaxNormalizer(Normalizer):
    def __init__(self, data_fit: np.ndarray):
        """data_fit : np.ndarray of shape (N, T) used to compute min and max"""
        self.data_min = data_fit.min(axis=1)
        self.data_max = data_fit.max(axis=1)
    
    def normalize(self, data):
        return (data - self.data_min[:, None]) / (self.data_max[:, None] - self.data_min[:, None])
    
    def denormalize(self, data):
        return data * (self.data_max[:, None] - self.data_min[:, None]) + self.data_min[:, None]
    

class NoNormalizer(Normalizer):
    def normalize(self, data):
        return data
    
    def denormalize(self, data):
        return data
    
def get_normalizer(name: str, data_fit: np.ndarray) -> Normalizer:
    if name == "zscore":
        return ZscoreNormalizer(data_fit)
    elif name == "minmax":
        return MinMaxNormalizer(data_fit)
    elif name == "none":
        return NoNormalizer()
    else:
        raise ValueError(f"Unknown normalizer: {name}, available are 'zscore', 'minmax', 'none'")