from typing import TYPE_CHECKING

from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch

from multi_time_gnn.utils import get_logger, register_model
from multi_time_gnn.test import prediction_step

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

log = get_logger()


def train_loop_mtgnn(model, dataset_train, dataset_val, config, optimizer=None, writer:"SummaryWriter"=None):
    """The training loop for mtgnn"""
    model.train()
    # nodes = [1:N] TODO implement when subgraphs
    train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=config.val_batch_size, shuffle=False)
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
    log.info(f"Starting Epoch 0: val: {(sum(log_loss_val) / len(log_loss_val)):.4f} - ref: {ref_value:.4f}")

    best_val_loss = float("inf")
    total_step_each_epoch = min(len(train_loader), config.nb_iter_per_epoch if config.nb_iter_per_epoch else len(train_loader))
    for i in range(config.n_epoch):
        log_loss = []
        model.train()
        nb_iter = 0
        for j, (x_train, y_train) in tqdm(enumerate(train_loader), total=total_step_each_epoch):
            # x_train of size BxNxT
            x_train = x_train.to(config.device)
            y_train = y_train.to(config.device)
            _, loss = model(x_train, y_train)
            nb_iter += 1
            if nb_iter > config.nb_iter_per_epoch and config.nb_iter_per_epoch:
                break
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Log
            log_loss.append(loss.item())
            if writer:
                writer.add_scalar('Loss/Train_Loss', loss.item(), j + i * total_step_each_epoch)


        _, loss_val = prediction_step(model, config, val_loader=val_loader, return_loss=True)
        if writer:
            writer.add_scalar("Loss/Val_Loss", loss_val, i * total_step_each_epoch)

        log.info(f"Epoch {i}: train: {(sum(log_loss) / len(log_loss)):.4f} - val: {loss_val:.4f} - ref: {ref_value:.4f}")

        # Log model with best val loss 
        if loss_val < best_val_loss :
            log.info(f"Save new model epoch {i}: val_loss ({loss_val:.4f}) < best_val_loss ({best_val_loss:.4f})")
            best_val_loss = loss_val
            register_model(model, config=config)


def train_loop_statistical(model, dataset_train, dataset_val, config, writer:"SummaryWriter"=None):
    """
    The training loop for the statistical model
    The goal is just to find the best lag for each channel for our local statistical model
    We only use validation to find the best lag
    """
    train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True)
    indices = torch.randperm(len(dataset_val))[:config.nb_val]
    dataset_val = Subset(dataset_val, indices)  # we take just a subset of the dataset to speed up the process
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False)
    nb_lags = config.lag_max - config.lag_min
    loss_val_with_lags = np.empty((nb_lags, config.N))

    log.info(f"Finding the best lag for our statistical model...")
    for nb_lag, lag in tqdm(enumerate(range(config.lag_min, config.lag_max))):
        loss_lag = np.zeros(config.N)
        for x_val, y_val in val_loader:
            x_val = x_val.to(config.device)
            y_val = y_val.to(config.device)
            _, loss = model(x_val, y_val, lag * np.ones(config.N, dtype=np.int8))
            loss_lag += loss
        loss_val_with_lags[nb_lag, :] = loss_lag
    # Keeping the best model
    model.best_lags = np.argmin(loss_val_with_lags, axis=0)
    log.info(f"Save new model")
    register_model(model, config=config)
