from datetime import datetime
from pathlib import Path
import torch
from torchinfo import summary

from multi_time_gnn.model import get_model
from multi_time_gnn.dataset import get_normalizer, read_dataset, TimeSeriesDataset, split_train_val_test
from multi_time_gnn.training import train_loop_mtgnn, train_loop_ar_local, train_loop_ar_global
from multi_time_gnn.visualization import pipeline_plotting
from multi_time_gnn.utils import get_tensorboard_writer, load_config, get_logger, load_model, register_model, set_all_global_seed, keep_config_model_kind
from multi_time_gnn.horizon import horizon_computing
from multi_time_gnn.preprocessing import preprocess_eeg
from multi_time_gnn.test import test_loss
from itertools import product
import shutil

def model_train_loop(config, dataset_train, dataset_val, normalizer):

    log = get_logger()
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
            input_size=(1, 1, config.N, config.timepoints_input),
            device=config.device
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    writer = get_tensorboard_writer(config)
    log.info(f"TensorBoard logging enabled in: {config.output_dir}/{config.tensorboard_dir}")

    val_loss = None
    log.info("Starting training...")
    try:
        if config.model_kind == "MTGNN":
            val_loss = train_loop_mtgnn(model, dataset_train, dataset_val, config, normalizer, optimizer, writer)
        elif config.model_kind == "AR_local":
            train_loop_ar_local(model, dataset_train, dataset_val, config, writer)
        elif config.model_kind == "AR_global":
            train_loop_ar_global(model, dataset_train, dataset_val, config)
    except KeyboardInterrupt:
        log.warning("Training interrupted by user.")

    return val_loss




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

    if config.model_kind == "MTGNN" and config.device == "auto":
        config.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )

    if config.model_kind == "MTGNN" and config.grid_search:
        log.info("Grid search enabled. Iterating over hyperparameters...")
        
        # Extract grid search parameters
        grid_params = {}
        for key, value in config.to_dict().items():
            if isinstance(value, list) and key not in ['list_horizon']:
                grid_params[key] = value
                log.info(f"Grid search parameter: {key} with values {value}")
        
        if not grid_params:
            log.warning("Grid search enabled but no list parameters found. Running single training.")
            val_loss = model_train_loop(config, dataset_train, dataset_val, normalizer)
        else:
            # Generate all combinations
            param_names = list(grid_params.keys())
            param_values = list(grid_params.values())
            combinations = list(product(*param_values))
            
            log.info(f"Total grid search combinations: {len(combinations)}")
            
            best_val_loss = float('inf')
            best_params = None
            
            for idx, combination in enumerate(combinations):
                log.info(f"\n{'='*50}")
                log.info(f"Grid search iteration {idx + 1}/{len(combinations)}")
                
                # Update config with current combination
                current_config = load_config()
                current_config = keep_config_model_kind(current_config)
                current_config.output_dir = str(dir_path / f"grid_{idx}")
                Path(current_config.output_dir).mkdir(parents=True, exist_ok=True)
                current_config.N = n_capteur
                
                for param_name, param_value in zip(param_names, combination):
                    setattr(current_config, param_name, param_value)
                    log.info(f"  {param_name}: {param_value}")
                
                # Train with current configuration
                val_loss = model_train_loop(current_config, dataset_train, dataset_val, normalizer)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = dict(zip(param_names, combination))
                    # Copy best model to main output directory
                    for file in Path(current_config.output_dir).glob("*"):
                        if file.is_file():  
                            shutil.copy(file, dir_path)
                
                log.info(f"Validation loss: {val_loss:.6f}")
            
            log.info(f"\n{'='*50}")
            log.info(f"Grid search completed!")
            log.info(f"Best validation loss: {best_val_loss:.6f}")
            log.info(f"Best parameters: {best_params}")
            
            # Update config with best parameters
            for param_name, param_value in best_params.items():
                setattr(config, param_name, param_value)
    else:
        val_loss = model_train_loop(config, dataset_train, dataset_val, normalizer)

    
    # Load best model
    log.info("Loading best model for evaluation...")
    best_model = load_model(get_model(config), Path(config.output_dir), config)
    best_model.to(config.device)

    log.info("Computing the test metrics...")
    test_loss(best_model, config, test, test_std, normalizer)

    if config.pipeline_plotting:
        log.info("Generating plots...")
        pipeline_plotting(best_model, test, normalizer, config)

    if config.plotting_horizon:
        log.info("Computing the horizon...")
        horizon_computing(best_model, test, config, normalizer, list_horizon=config.list_horizon)

    log.info("✅ Pipeline completed successfully")
