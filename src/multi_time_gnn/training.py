from typing import TYPE_CHECKING

from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from multi_time_gnn.utils import get_logger, register_model
from multi_time_gnn.test import prediction_step

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

log = get_logger()


def train_loop(model, dataset_train, dataset_val, optimizer, config, writer:"SummaryWriter"=None):
    model.train()
    # nodes = [1:N] TODO implement when subgraphs
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
    for i in range(config.n_epoch):
        # Reshuffle the training data each epoch
        train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)
        total_step_each_epoch = min(len(train_loader), config.nb_iter_per_epoch if config.nb_iter_per_epoch else len(train_loader))

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
            writer.add_scalar("Loss/Val_Loss", loss_val, (i + 1) * total_step_each_epoch)

        log.info(f"Epoch {i}: train: {(sum(log_loss) / len(log_loss)):.4f} - val: {loss_val:.4f} - ref: {ref_value:.4f}")

        # Log model with best val loss 
        if loss_val < best_val_loss :
            log.info(f"Save new model epoch {i}: val_loss ({loss_val:.4f}) < best_val_loss ({best_val_loss:.4f})")
            best_val_loss = loss_val
            register_model(model, config=config)
            