from dataset import get_batch
from utils import get_logger

log = get_logger()


def train_lopp(model, dataset, optimizer, config):
    model.train()
    log_loss = []
    # nodes = [1:N] TODO implement when subgraphs
    for i in range(1, config.n_epoch + 1):
        x, y = get_batch(config.batch_size, dataset, config.timepoints_input)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()

        # Log
        log_loss.append(loss.item())
        if i % config.log_each == 0:
            log.info(f"Step {i}: {sum(log_loss[-config.log_each:]) / config.log_each}")
