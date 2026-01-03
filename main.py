from datetime import datetime
from pathlib import Path
import torch
from torchinfo import summary

from multi_time_gnn.model import get_model
from multi_time_gnn.dataset import get_normalizer, read_dataset, TimeSeriesDataset, split_train_val_test
from multi_time_gnn.training import train_loop_mtgnn, train_loop_statistical
from multi_time_gnn.visualization import pipeline_plotting
from multi_time_gnn.utils import get_tensorboard_writer, load_config, get_logger, load_model, register_model, set_all_global_seed, keep_config_model_kind
from multi_time_gnn.horizon import horizon_computing
from multi_time_gnn.preprocessing import preprocess_eeg
from multi_time_gnn.test import test_loss


if __name__ == "__main__":

    config = load_config()
    config = keep_config_model_kind(config)
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_path = Path("saved_models/" + now)
    dir_path.mkdir(parents=True, exist_ok=True)
    config.output_dir = str(dir_path)


    log = get_logger("main", config.log_level, path_file=str(dir_path / "training.log"))
    log.debug(config)
    log.info(f"Model used '{config.model_kind}'")

    set_all_global_seed(config.seed)

    dataset = read_dataset(config.dataset_name, path_eeg=config.path_eeg)  # NxT

    if config.preprocessing and config.dataset_name == "eeg":
        log.info("Preprocessing EEG data: high-pass filtering and resampling...")
        dataset = preprocess_eeg(dataset, config)
    elif config.preprocessing:
        log.warning("Preprocessing is only implemented for EEG dataset currently.")
    

    n_capteur, nb_timestamp = dataset.shape
    train, val, test = split_train_val_test(dataset, train_ratio=config.train_ratio, val_ratio=(1 - config.train_ratio)/2)
    val_std, test_std = val.std(), test.std()
    normalizer = get_normalizer(config.normalization_method, train)
    train = normalizer.normalize(train)
    val = normalizer.normalize(val)
    test = normalizer.normalize(test)
    dataset_train, dataset_val = TimeSeriesDataset(train, config), TimeSeriesDataset(val, config)
    log.info(f"Dataset '{config.dataset_name}' shape : {n_capteur, nb_timestamp}")
    # Update config with dataset specific parameters
    config.N = n_capteur

    model_class = get_model(config)
    model = model_class(config)
    if config.model_kind == "MTGNN":
        if config.device == "auto":
            config.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
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
        if config.model_kind == "MTGNN":
            train_loop_mtgnn(model, dataset_train, dataset_val, config, normalizer, optimizer, writer)
        elif config.model_kind == "AR_local":
            train_loop_statistical(model, dataset_train, dataset_val, config, writer)
    except KeyboardInterrupt:
        log.warning("Training interrupted by user.")

    # Load best model
    log.info("Loading best model for evaluation...")
    best_model = load_model(get_model(config), Path(config.output_dir), config)
    best_model.to(config.device)

    log.info("Computing the test metrics...")
    test_loss(model, config, test, test_std, normalizer)

    if config.pipeline_plotting:
        log.info("Generating plots...")
        pipeline_plotting(best_model, test, normalizer, config)

    if config.plotting_horizon:
        log.info("Computing the horizon...")
        horizon_computing(best_model, test, config, normalizer, list_horizon=config.list_horizon)

    log.info("✅ Pipeline completed successfully")
