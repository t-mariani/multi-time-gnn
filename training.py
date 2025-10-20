from dataset import get_batch


def train_lopp(model, dataset, optimizer, config):

    log_loss = []
    # nodes = [1:N] TODO implement when subgraphs
    for i in range(1, config.n_epoch + 1):
        x, y = get_batch(config.batch_size, dataset)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()

        # Log
        log_loss.append(loss.item())
        if i % config.log_each == 0:
            print(f"Step {i}: {log_loss[-config.log_each:] / config.log_each}")
