from datetime import datetime
from pathlib import Path
import torch
from torchinfo import summary

from multi_time_gnn.model import NextStepModel
from multi_time_gnn.dataset import normalize, read_dataset, find_mean_std, TimeSeriesDataset
from multi_time_gnn.training import train_loop
from multi_time_gnn.test import test_step
from multi_time_gnn.visualization import pipeline_plotting
from multi_time_gnn.utils import load_config, get_logger, register_model


if __name__ == "__main__":

    config = load_config()
    log = get_logger("main", config.log_level)
    log.debug(config)
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = read_dataset(config.dataset_name)  # NxT
    nb_timestep = dataset.shape[-1]
    train_size = int(nb_timestep * config.train_ratio)
    val_size = int(nb_timestep * (1 - config.train_ratio) // 2)
    train, val, test = (
        dataset[:, :train_size],
        dataset[:, train_size:train_size+val_size],
        dataset[:, train_size+val_size:]
        )

    y_mean, y_std = find_mean_std(train)
    train = normalize(train, y_mean, y_std)
    val = normalize(val, y_mean, y_std)
    test = normalize(test, y_mean, y_std)
    dataset_train, dataset_val = TimeSeriesDataset(train, config), TimeSeriesDataset(val, config)
    N, T = dataset.shape
    log.info(f"Dataset '{config.dataset_name}' shape : {N, T}")
    # Update config with dataset specific parameters
    config.N = N
    # Update config with output directory with timestamp
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_path = Path("saved_models/" + now)
    config.output_dir = str(dir_path)

    model = NextStepModel(config)
    model.to(config.device)

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
        train_loop(model, dataset_train, dataset_val, optimizer, config)
    except KeyboardInterrupt:
        log.warning("Training interrupted by user.")

    log.info("Registering model...")
    register_model(model, config=config)

    # log.info("Testing model...")
    #ypred = test_step(model, test, config) # commented because plotting function already does testing

    log.info("Generating plots...")
    pipeline_plotting(model, test, config)

    log.info("✅ Pipeline completed successfully")
