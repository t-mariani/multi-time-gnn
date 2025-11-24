from typing import Literal
from einops import rearrange
import torch
import numpy as np
from multi_time_gnn.utils import get_logger
from torch.utils.data import Dataset

log = get_logger()

available_datasets = ["electricity", "traffic", "solar", "exchange"]


def find_mean_std(train) -> tuple[np.ndarray, np.ndarray]:
    """Find the mean and the standard devitation for all the dimension of the training"""
    return train.mean(axis=1), train.std(axis=1)


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


def get_batch(
    batch_size: int, dataset: np.ndarray, t, y_t=1, index=None, device='cpu'
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    batch_size : int
    dataset : array size (N, bigT)
    t : int, input timepoints
    y_t : int, output timepoints
    index : list of int, optional, timepoints indices for the batch

    Returns :
    x : torch.Tensor of size (batch_size,1,N,t)
    y : torch.Tensor of size (batch_size,1,N,y_t)
    """
    N, bigT = dataset.shape
    if index is not None:
        idx = index
    else:
        idx = torch.randint(0, bigT - t - y_t, (batch_size,))
    x = torch.Tensor(np.array([dataset[:, i: i + t] for i in idx]))
    y = torch.Tensor(np.array([dataset[:, i + t: i + t + y_t] for i in idx]))
    return x[:, None, :, :].to(device), y[:, None, :, :].to(device)  # Return : Bx1xNxT, Bx1xNxy_t


class TimeSeriesDataset(Dataset):
    def __init__(self, data, config):
        self.data = data
        self.nb_points = config.timepoints_input
        self.device = config.device

    def __len__(self):
        return self.data.shape[-1] - self.nb_points - 1

    def __getitem__(self, idx):
        x = self.data[:, idx:idx + self.nb_points]
        y = self.data[:, idx + self.nb_points]
        x = x[None, :, :]  # 1xNxT
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def normalize(data, mean, std):
    """Normalize the data over all the dimension with mean and std"""
    return (data - mean[:, None]) / std[:, None]


def denormalize(data, mean, std):
    """Denormalize the data over all the dimension with mean and std"""
    return data * std[:, None] + mean[:, None]
