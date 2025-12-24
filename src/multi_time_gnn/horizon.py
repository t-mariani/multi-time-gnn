
from einops import rearrange
from multi_time_gnn.utils import get_logger
from multi_time_gnn.dataset import TimeSeriesDataset, denormalize
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from tqdm import tqdm

from multi_time_gnn.visualization import plot_prediction_horizons


log = get_logger()

def horizon_computing(model, test, config, y_mean, y_std, list_horizon=None):
    """Compute the horizon for 3 - 6 - 12 - 24 timestep"""
    if list_horizon is None:
        list_horizon = [3, 6, 12, 24]
    max_horizon = max(list_horizon)
    
    dataset_test = TimeSeriesDataset(test, config, max_horizon)
    dataloader_test = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False)
    loss_result_denorm = torch.zeros(config.nb_test, max_horizon)
    loss_result_norm = torch.zeros(config.nb_test, max_horizon)
    prediction_result_list = []
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
            prediction_result_list.append(x[:, :, :, -max_horizon:])

            if k >= config.nb_test - 1:
                break
    loss_horizon_norm = loss_result_norm.mean(dim=0)
    loss_horizon_denorm = loss_result_denorm.mean(dim=0)
    string_norm = "Horizon Normalized MSE: "
    string_denorm = "Horizon Denormalized MSE: "
    for h in list_horizon:
        string_norm += f"{h}: {loss_horizon_norm[h-1].item():.3f}  "
        string_denorm += f"{h}: {loss_horizon_denorm[h-1].item():.3f}  "
    log.info(string_norm)
    log.info(string_denorm)

    # Plotting
    prediction_result = torch.cat(prediction_result_list, dim=0).to("cpu").numpy()  # Can be seen has shape T=nb_test,1,N,max_horizon
    signal_horizons = rearrange(prediction_result, "T 1 N H -> H N T")
    signal_horizons_denorm = denormalize(signal_horizons, y_mean, y_std)
    cliped_capteur = min(10, config.N)  # To avoid too much plots
    test = test[:cliped_capteur, :]
    signal_horizons_denorm = signal_horizons_denorm[:, :cliped_capteur, :]
    plot_prediction_horizons(test, signal_horizons_denorm, selected_horizons=list_horizon)
