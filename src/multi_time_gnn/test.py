import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from multi_time_gnn.dataset import TimeSeriesDataset
from multi_time_gnn.utils import get_logger

log = get_logger()

def prediction_step(model, config, val=None, val_loader=None, return_loss=False):
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
        log.info("Run prediction step")
        for x_val, y_val in tqdm(val_loader):
            x_val = x_val.to(config.device)
            y_val = y_val.to(config.device)
            prediction, loss = model(x_val, y_val)
            loss_val.append(loss.item())
            all_prediction.append(prediction)
        total_ypred = torch.cat(all_prediction, dim=0)
    total_ypred = total_ypred.squeeze().T # From Bx1xNx1 -> NxB because here B=T

    if return_loss:
        return total_ypred, sum(loss_val) / len(loss_val)
    return total_ypred # NxT 


def predict_multi_step(model, input_sequence, n_steps):
    model.eval()
    input_seq = input_sequence.clone().detach()
    predictions = []
    with torch.no_grad():
        log.info("Multi-step prediction:")
        for _ in tqdm(range(n_steps)):
            y_pred, _ = model(input_seq)
            predictions.append(y_pred)

            # Update the input sequence by appending the new prediction and removing the oldest time point
            input_seq = torch.cat((input_seq[:, :, :, 1:], y_pred), dim=-1)

    predictions = torch.cat(predictions, axis=0).cpu().squeeze().numpy()
    log.debug(f"Predictions shape in multi-step: {predictions.shape}")
    return predictions.T  # Shape (N, n_steps)
