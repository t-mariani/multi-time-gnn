from typing import Literal
import torch
import numpy as np

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


def get_batch(batch_size, dataset, t):
    bigT, N = dataset.shape
    idx = torch.randint(0, bigT, (batch_size,))
    x = torch.Tensor(np.array([dataset[i : i + t, :] for i in idx]))
    y = torch.Tensor(np.array([dataset[i + t, :] for i in idx])).reshape(
        (batch_size, 1, N)
    )

    return x, y  # Return : BxTxN, Bx1xN
