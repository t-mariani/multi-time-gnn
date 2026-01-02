import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import json

from multi_time_gnn.dataset import TimeSeriesDataset
from multi_time_gnn.utils import get_logger

log = get_logger()

def test_loss(model, config, test_data, test_std, normalizer):
    """Compute the RSE of the dataset"""
    dataset_test = TimeSeriesDataset(test_data, config)
    test_loader = DataLoader(dataset_test, batch_size=config.val_batch_size, shuffle=False)
    model.eval()
    with torch.no_grad():
        y_pred_tot_norm = torch.tensor([])
        y_true_tot_norm = torch.tensor([])
        for x_test, y_test in tqdm(test_loader):
            x_test = x_test.to(config.device)
            y_test = y_test.to(config.device)
            y_predict, loss = model(x_test, y_test)
            y_pred_tot_norm = torch.concat((y_pred_tot_norm, y_predict.to("cpu")))
            y_true_tot_norm = torch.concat((y_true_tot_norm, y_test.to("cpu")))
    y_pred_tot_norm = y_pred_tot_norm.squeeze()
    y_pred_tot_denorm = normalizer.denormalize(y_pred_tot_norm.T).T
    y_true_tot_denorm = normalizer.denormalize(y_true_tot_norm.T).T
    rse = _compute_rse(y_pred_tot_denorm, y_true_tot_denorm)
    corr = _compute_corr(y_pred_tot_denorm, y_true_tot_denorm)
    log.info(f"Horizon: {config.horizon_prediction} - RSE: {rse:.4f} - Corr: {corr:.4f}")
    with open(f"{config.output_dir}/results.json", 'w') as f:
        json.dump({
            "horizon": config.horizon_prediction,
            "RSE": rse.item(),
            "CORR": corr.item()
        }, f)

def _compute_rse(y_pred, y_true):
    """Compute the root relative squared error"""
    mse = F.mse_loss(y_pred, y_true) 
    numerator = torch.sqrt(mse)
    denominator = y_true.std(unbiased=False)
    rse = numerator / (denominator + 1e-7)
    return rse


def _compute_corr(y_pred, y_true):
    """
    Compute the Empirical Correlation Coefficient (CORR).
    """
    mean_p = y_pred.mean(dim=0)
    mean_t = y_true.mean(dim=0)
    std_p = y_pred.std(dim=0, unbiased=False)
    std_t = y_true.std(dim=0, unbiased=False)
    covariance = ((y_pred - mean_p) * (y_true - mean_t)).mean(dim=0)
    correlation = covariance / (std_p * std_t + 1e-7)
    return correlation.mean()


def prediction_step(model, config, normalizer, val=None, val_loader=None, return_loss=False):
    """Run a test step on the model with no gradient and backwward prop

    Parameters:
        model: torch.nn.Module
        config: Box, configuration file 
        val: numpy array of values (NxT), will be batched (config.val_batch_size)
        val_loader: torch.DataLoader 
        return_loss: bool, return a tuple with prediction, loss
    """
    if val_loader is None:
        val_dataset = TimeSeriesDataset(val, config)
        val_loader = DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False)

    model.eval()
    with torch.no_grad():
        loss_val = []
        all_prediction = []
        all_expected = []
        log.info("Run prediction step")
        for x_val, y_val in tqdm(val_loader):
            x_val = x_val.to(config.device)
            y_val = y_val.to(config.device)
            prediction, loss = model(x_val, y_val)
            loss_val.append(loss.item())
            all_prediction.append(prediction)
            all_expected.append(y_val)
        total_ypred = torch.cat(all_prediction, dim=0)
        total_yexpected = torch.cat(all_expected, dim=0)
    total_ypred = total_ypred.squeeze().T # From Bx1xNx1 -> NxB because here B=T
    total_yexpected = total_yexpected.squeeze().T # From Bx1xNx1 -> NxB because here B=T
    total_yexpected_denorm = normalizer.denormalize(total_yexpected.cpu()).T
    total_ypred_denorm = normalizer.denormalize(total_ypred.cpu()).T
    if return_loss:
        return total_ypred, sum(loss_val) / len(loss_val), _compute_rse(total_ypred_denorm, total_yexpected_denorm), _compute_corr(total_ypred_denorm, total_yexpected_denorm)
    return total_ypred # NxT 


def predict_multi_step(model, input_sequence, n_steps):
    model.eval()
    input_seq = input_sequence.clone().detach()
    predictions = []
    with torch.no_grad():
        log.info("Multi-step prediction:")
        for _ in tqdm(range(n_steps)):
            y_pred, _ = model(input_seq)

            # for the statistical model
            if len(y_pred.shape) == 1:
                y_pred = y_pred[None, None, :, None]

            predictions.append(y_pred)

            # Update the input sequence by appending the new prediction and removing the oldest time point
            input_seq = torch.cat((input_seq[:, :, :, 1:], y_pred), dim=-1)

    predictions = torch.cat(predictions, axis=0).cpu().squeeze().numpy()
    log.debug(f"Predictions shape in multi-step: {predictions.shape}")
    return predictions.T  # Shape (N, n_steps)
