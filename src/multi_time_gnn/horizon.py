
from multi_time_gnn.utils import get_logger
from multi_time_gnn.dataset import TimeSeriesDataset, denormalize
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from tqdm import tqdm


log = get_logger()

def horizon_computing(model, test, config, y_mean, y_std, max_horizon=24):
    """Compute the horizon for 3 - 6 - 12 - 24 timestep"""
    dataset_test = TimeSeriesDataset(test, config, max_horizon)
    dataloader_test = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False)
    loss_result_denorm = torch.zeros(config.nb_test, max_horizon)
    loss_result_norm = torch.zeros(config.nb_test, max_horizon)
    model.eval()
    with torch.no_grad():
        for k, (x, y) in tqdm(enumerate(dataloader_test)):
            for j in range(max_horizon):
                x = x.to(config.device)
                y_true = y[:, :, j].to(config.device)
                y_pred, loss_norm = model(x, y_true)
                # without normalization
                y_denorm_pred = denormalize(y_pred.squeeze().to("cpu").T, y_mean, y_std).T
                y_denorm_true = denormalize(y_true.squeeze().to("cpu").T, y_mean, y_std).T
                loss_result_denorm[k, j] = F.mse_loss(y_denorm_pred, y_denorm_true)
                loss_result_norm[k, j] = loss_norm
                # we use the prediction to guess the next step
                x = torch.concat((x[:, :, :, :-1], y_pred), dim=-1) 
            if k >= config.nb_test - 1:
                break
    loss_horizon_norm = loss_result_norm.mean(dim=0)
    loss_horizon_denorm = loss_result_denorm.mean(dim=0)
    log.info(f"Result horizon normalized: 3: {loss_horizon_norm[2].item():.3f} - 3: {loss_horizon_norm[5].item():.3f} - 12: {loss_horizon_norm[11].item():.3f} - 24: {loss_horizon_norm[23].item():.3f}")
    log.info(f"Result horizon denormalized: 3: {loss_horizon_denorm[2].item():.3f} - 3: {loss_horizon_denorm[5].item():.3f} - 12: {loss_horizon_denorm[11].item():.3f} - 24: {loss_horizon_denorm[23].item():.3f}")
