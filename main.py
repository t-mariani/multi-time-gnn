import torch

from model import NextStepModel
from dataset import read_dataset
from training import train_lopp
from utils import load_config

if __name__ == "__main__":
    config = load_config()
    dataset = read_dataset("solar")
    T, N = dataset.shape
    config.N = N
    model = NextStepModel(config)

    optimizer = torch.optim.AdamW(
        model.parameter(),
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_lopp(model, dataset, optimizer, config)
