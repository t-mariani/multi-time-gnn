from pathlib import Path

from multi_time_gnn.dataset import get_normalizer, read_dataset, split_train_val_test
from multi_time_gnn.horizon import horizon_computing
from multi_time_gnn.model import NextStepModel
from multi_time_gnn.visualization import pipeline_plotting
from multi_time_gnn.utils import get_logger, load_config, get_latest_dir, load_model


if __name__ == "__main__":
    res_dir = "saved_models"

    # Get latest model directory, change if you want to load another model
    model_dir = get_latest_dir(Path(res_dir))
    # Load config, model and dataset
    config = load_config(model_dir / "config.yaml")
    model = load_model(NextStepModel, model_dir, config)
    model.to(config.device)
    dataset = read_dataset(config.dataset_name, config.path_eeg)

    log = get_logger("main", config.log_level)
    log.debug(config)

    train, _, test = split_train_val_test(dataset, train_ratio=config.train_ratio, val_ratio=(1 - config.train_ratio)/2)

    normalizer = get_normalizer(config.normalization_method, train)
    test = normalizer.normalize(test)

    log.info("Generating plots...")
    pipeline_plotting(model, test, normalizer, config)

    log.info("Computing the horizon...")
    horizon_computing(model, test, config, normalizer, list_horizon=config.list_horizon)
