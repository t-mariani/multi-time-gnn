import torch
from torchinfo import summary

from multi_time_gnn.model import NextStepModel
from multi_time_gnn.dataset import normalize, read_dataset
from multi_time_gnn.training import train_loop
from multi_time_gnn.utils import load_config, get_logger


if __name__ == "__main__":

    config = load_config()
    log = get_logger("main", config.log_level)
    log.debug(config)

    dataset = read_dataset(config.dataset_name)
    dataset = normalize(dataset)
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

    train_loop(model, dataset, optimizer, config)
