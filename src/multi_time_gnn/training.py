from multi_time_gnn.utils import get_logger
from multi_time_gnn.test import loss_test_step
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from tqdm import tqdm

log = get_logger()


def train_loop(model, dataset_train, dataset_val, optimizer, config):
    model.train()
    # nodes = [1:N] TODO implement when subgraphs
    train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=False)
    model.eval()
    log_loss_val = []
    log_loss_ref = []
    for x_val, y_val in val_loader:
        x_val = x_val.to(config.device)
        y_val = y_val.to(config.device)
        _, loss = model(x_val, y_val)
        log_loss_val.append(loss.item())
        log_loss_ref.append(F.mse_loss(x_val.squeeze()[:, :, -1], y_val))
    ref_value = sum(log_loss_ref) / len(log_loss_ref)
    log.info(f"Step 0: val: {(sum(log_loss_val) / len(log_loss_val)):.3f} - ref: {ref_value:.3f}")
    for i in range(1, config.n_epoch + 1):
        log_loss = []
        model.train()
        nb_iter = 0
        for x_train, y_train in train_loader:
            # x_train of size BxNxT
            x_train = x_train.to(config.device)
            y_train = y_train.to(config.device)
            _, loss = model(x_train, y_train)
            nb_iter += 1
            if nb_iter > config.nb_iter_per_epoch:
                break
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Log
            log_loss.append(loss.item())
        loss_val = loss_test_step(model, config, val_loader=val_loader)

        if i % config.log_each == 0:
            log.info(f"Step {i}: train: {(sum(log_loss) / len(log_loss)):.3f} - val: {loss_val:.3f} - ref: {ref_value:.3f}")
