import torch
from utils import get_logger
from torchinfo import summary

from model import NextStepModel
from dataset import read_dataset
from training import train_lopp
from utils import load_config


if __name__ == "__main__":

    config = load_config()
    log = get_logger("main", config.log_level)
    log.debug(config)

    dataset = read_dataset(config.dataset_name)
    T, N = dataset.shape
    log.info(f"Dataset '{config.dataset_name}' shape : {T, N}")
    config.N = N
    model = NextStepModel(config)

    summary(
        model,
        depth=5,
        input_size=(1, 1, N, config.timepoints_input),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_lopp(model, dataset, optimizer, config)
