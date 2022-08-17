import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import madgrad
from train_utils import train_loop_regular, eval_loop_regular, train_loop_wtih_metadata, eval_loop_wtih_metadata

def train(train_dataset, val_dataset, test_dataset, model, config):
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    criterion = CrossEntropyLoss()
    if config.optimizer_params.optimizer == "madgrad":
        optimizer = madgrad.MADGRAD(model.parameters(), lr=config.optimizer_params.optimizer)
    elif config.optimizer_params.optimizer == "Adam":
        optimizer = Adam(model.parameters(), lr=config.optimizer_params.optimizer)
    else:
        raise Exception("Optimizer {config.optimizer_params.optimizer} not supported")

    if config.metadata_in_dataset:
        train_loop = train_loop_wtih_metadata
        eval_loop = eval_loop_wtih_metadata
    else:
        train_loop = train_loop_regular
        eval_loop = eval_loop_regular

    device = torch.device(config.device)

    losses = []
    metrics = []
    model = model.to(device)

    for epoch in tqdm(range(config.n_epochs)):
        loss = train_loop(model, train_loader, device, criterion, optimizer)

        metric = eval_loop(model, val_loader, device)

        losses.append(loss)
        metrics.append(metric)
        print(f'Epoch {epoch} complete, loss = {loss}, metrics: {metric}')

    test_metrics = eval_loop(model, test_loader, device)

    print('Finished Training')
    return losses, metrics, test_metrics