from pathlib import Path

from multi_time_gnn.dataset import get_normalizer, read_dataset, split_train_val_test
from multi_time_gnn.horizon import horizon_computing
from multi_time_gnn.model import get_model
from multi_time_gnn.preprocessing import preprocess_eeg
from multi_time_gnn.test import test_loss
from multi_time_gnn.visualization import pipeline_plotting
from multi_time_gnn.utils import get_logger, load_config, get_latest_dir, load_model


if __name__ == "__main__":
    res_dir = "saved_models"

    # Get latest model directory, change if you want to load another model
    model_dir = get_latest_dir(Path(res_dir))
    # Load config, model and dataset
    config = load_config(model_dir / "config.yaml")
    log = get_logger("main", config.log_level)
    log.debug(config)
    
    model_class = get_model(config)
    model = load_model(model_class, model_dir, config)
    model.to(config.device)
    dataset = read_dataset(config.dataset_name, config.path_eeg)

    if config.preprocessing and config.dataset_name == "eeg":
        log.info("Preprocessing EEG data: high-pass filtering and resampling...")
        dataset = preprocess_eeg(dataset, config)
    elif config.preprocessing:
        log.warning("Preprocessing is only implemented for EEG dataset currently.")


    train, _, test = split_train_val_test(dataset, train_ratio=config.train_ratio, val_ratio=(1 - config.train_ratio)/2)

    normalizer = get_normalizer(config.normalization_method, train)
    test = normalizer.normalize(test)

    log.info("Computing the test metrics...")
    test_loss(model, config, test, test.std(), normalizer)

    if config.pipeline_plotting:
        log.info("Generating plots...")
        pipeline_plotting(model, test, normalizer, config)

    if config.plotting_horizon:
        log.info("Computing the horizon...")
        horizon_computing(model, test, config, normalizer, list_horizon=config.list_horizon)