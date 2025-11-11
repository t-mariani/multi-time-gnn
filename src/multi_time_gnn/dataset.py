from typing import Literal
from einops import rearrange
import torch
import numpy as np
from multi_time_gnn.utils import get_logger

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
    )  # Return size : TxN # TODO return tensor instead to speed up get_batch


def get_batch(
    batch_size: int, dataset: np.ndarray, t, y_t=1, index=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    batch_size : int
    dataset : array size (bigT,N)
    t : int, input timepoints
    y_t : int, output timepoints
    index : list of int, optional, timepoints indices for the batch

    Returns :
    x : torch.Tensor of size (batch_size,1,N,t)
    y : torch.Tensor of size (batch_size,1,N,y_t)
    """
    bigT, N = dataset.shape
    if index is not None:
        idx = index
    else:
        idx = torch.randint(0, bigT - t - y_t, (batch_size,))
    x = rearrange(
        torch.Tensor(np.array([dataset[i : i + t, :] for i in idx])), "b t n -> b 1 n t"
    )
    y = rearrange(
        torch.Tensor(np.array([dataset[i + t : i + t + y_t, :] for i in idx])),
        "b t n -> b 1 n t",
    )
    return x, y  # Return : Bx1xNxT, Bx1xNxy_t


def normalize(data):
    print(np.max(np.abs(data), axis=0).shape)
    res = data / np.max(np.abs(data), axis=0, keepdims=True, initial=1e-7)
    return res
