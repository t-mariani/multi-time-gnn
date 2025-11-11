from pathlib import Path

from multi_time_gnn.dataset import read_dataset, normalize
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
    dataset = read_dataset(config.dataset_name)

    log = get_logger("main", config.log_level)
    log.debug(config)

    train_size = int(len(dataset) * config.train_ratio)
    train, val = dataset[:train_size], dataset[train_size:]
    val = normalize(val)

    log.info("Generating plots...")
    pipeline_plotting(model, val, config)
