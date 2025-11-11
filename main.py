from datetime import datetime
from pathlib import Path
import torch
from torchinfo import summary

from multi_time_gnn.model import NextStepModel
from multi_time_gnn.dataset import normalize, read_dataset
from multi_time_gnn.training import train_loop
from multi_time_gnn.test import test_step
from multi_time_gnn.visualization import pipeline_plotting
from multi_time_gnn.utils import load_config, get_logger, register_model


if __name__ == "__main__":

    config = load_config()
    log = get_logger("main", config.log_level)
    log.debug(config)

    dataset = read_dataset(config.dataset_name)
    train_size = int(len(dataset) * config.train_ratio)
    train, val = dataset[:train_size], dataset[train_size:]

    train = normalize(train)
    val = normalize(val)  # weird
    T, N = dataset.shape
    log.info(f"Dataset '{config.dataset_name}' shape : {T, N}")
    # Update config with dataset specific parameters
    config.N = N
    # Update config with output directory with timestamp
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_path = Path("saved_models/" + now)
    config.output_dir = str(dir_path)

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

    log.info("Starting training...")
    try:
        train_loop(model, train, optimizer, config)
    except KeyboardInterrupt:
        log.warning("Training interrupted by user.")

    log.info("Registering model...")
    register_model(model, config=config)

    # log.info("Testing model...")
    # ypred = test_step(model, val, config) # commented because plotting function already does testing

    log.info("Generating plots...")
    pipeline_plotting(model, val, config)

    log.info("✅ Pipeline completed successfully")
