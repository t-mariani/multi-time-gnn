from datetime import datetime
from pathlib import Path
import torch
from torchinfo import summary

from multi_time_gnn.model import NextStepModel
from multi_time_gnn.dataset import get_normalizer, read_dataset, TimeSeriesDataset, split_train_val_test
from multi_time_gnn.training import train_loop
from multi_time_gnn.visualization import pipeline_plotting
from multi_time_gnn.utils import get_tensorboard_writer, load_config, get_logger, load_model, register_model, set_all_global_seed
from multi_time_gnn.horizon import horizon_computing


if __name__ == "__main__":

    config = load_config()
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_path = Path("saved_models/" + now)
    dir_path.mkdir(parents=True, exist_ok=True)
    config.output_dir = str(dir_path)

    if config.device == "auto":
        config.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

    log = get_logger("main", config.log_level, path_file=str(dir_path / "training.log"))
    log.debug(config)

    set_all_global_seed(config.seed)

    dataset = read_dataset(config.dataset_name)  # NxT
    n_capteur, nb_timestamp = dataset.shape
    train, val, test = split_train_val_test(dataset, train_ratio=config.train_ratio, val_ratio=(1 - config.train_ratio)/2)

    normalizer = get_normalizer(config.normalization_method, train)
    train = normalizer.normalize(train)
    val = normalizer.normalize(val)
    test = normalizer.normalize(test)
    dataset_train, dataset_val = TimeSeriesDataset(train, config), TimeSeriesDataset(val, config)
    log.info(f"Dataset '{config.dataset_name}' shape : {n_capteur, nb_timestamp}")
    # Update config with dataset specific parameters
    config.N = n_capteur

    model = NextStepModel(config)
    model.to(config.device)

    summary(
        model,
        depth=5,
        input_size=(1, 1, n_capteur, config.timepoints_input),
        device=config.device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    writer = get_tensorboard_writer(config)
    log.info(f"TensorBoard logging enabled in: {config.output_dir}/{config.tensorboard_dir}")

    log.info("Starting training...")
    try:
        train_loop(model, dataset_train, dataset_val, optimizer, config, writer)
    except KeyboardInterrupt:
        log.warning("Training interrupted by user.")

    # Load best model
    log.info("Loading best model for evaluation...")
    best_model = load_model(NextStepModel, Path(config.output_dir), config)
    best_model.to(config.device)

    log.info("Generating plots...")
    pipeline_plotting(best_model, test, normalizer, config)

    log.info("Computing the horizon...")
    horizon_computing(best_model, test, config, normalizer, list_horizon=config.list_horizon)

    log.info("✅ Pipeline completed successfully")
