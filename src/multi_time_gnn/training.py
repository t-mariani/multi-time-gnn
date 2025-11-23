from multi_time_gnn.dataset import get_batch
from multi_time_gnn.utils import get_logger
from torch.utils.data import DataLoader
from tqdm import tqdm

log = get_logger()


def train_loop(model, dataset_train, dataset_val, optimizer, config):
    model.train()
    # nodes = [1:N] TODO implement when subgraphs
    train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=False)
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
            # Log
            log_loss.append(loss.item())
        model.eval()
        log_loss_val = []
        for x_val, y_val in val_loader:
            x_val = x_val.to(config.device)
            y_val = y_val.to(config.device)
            _, loss = model(x_val, y_val)
            log_loss_val.append(loss.item())
        if i % config.log_each == 0:
            log.info(f"Step {i}: train: {(sum(log_loss) / len(log_loss)):.3f} - val: {(sum(log_loss_val) / len(log_loss_val)):.3f}")
