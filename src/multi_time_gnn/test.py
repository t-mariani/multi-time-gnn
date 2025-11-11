import torch
from tqdm import tqdm
import numpy as np

from multi_time_gnn.dataset import get_batch
from multi_time_gnn.utils import get_logger

log = get_logger()


def test_step(model, val, config):
    n_possible_validation = len(val) - config.timepoints_input - 1  # len(val) = T_val

    model.eval()
    with torch.no_grad():
        total_ypred = []
        losses = []
        for n in tqdm(range(0, n_possible_validation, config.batch_size)):
            index = range(n, min(n + config.batch_size, n_possible_validation))

            xt, yt = get_batch(
                n_possible_validation,
                val,
                config.timepoints_input,
                y_t=1,
                index=index,
            )
            y_pred, loss = model(xt, yt)
            total_ypred.append(y_pred.cpu())
            losses.append(loss.item())
    total_ypred = torch.cat(total_ypred, dim=0).squeeze()

    log.info(f"Test Loss: {np.mean(losses):.4f}")
    log.debug(f"total_ypred length: {total_ypred.shape}")
    return total_ypred.numpy()  # Shape (n_possible_validation, N)


def predict_multi_step(model, input_sequence, n_steps):
    model.eval()
    input_seq = input_sequence.clone().detach()

    predictions = []
    with torch.no_grad():
        for _ in tqdm(range(n_steps)):
            y_pred, _ = model(input_seq)
            predictions.append(y_pred.cpu().squeeze().numpy())

            # Update the input sequence by appending the new prediction and removing the oldest time point
            input_seq = torch.cat((input_seq[:, :, :, 1:], y_pred), dim=-1)

    predictions = np.array(predictions)
    log.debug(f"Predictions shape in multi-step: {predictions.shape}")
    return predictions  # Shape (n_steps, N)
