import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def train_loop_regular(model, dataloader, device, criterion, optimizer):
    running_loss = 0.0
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        input_ids = torch.stack(batch['input_ids'], dim=1).to(device)
        token_type_ids = torch.stack(batch['token_type_ids'], dim=1).to(device)
        attention_mask = torch.stack(batch['attention_mask'], dim=1).to(device)

        labels = batch['label'].to(device)

        outputs = model(input_ids, token_type_ids, attention_mask)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
    return running_loss / len(dataloader)

def eval_loop_regular(model, dataloader, device):
    preds = []
    targets = []
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader)):
            input_ids = torch.stack(batch['input_ids'], dim=1).to(device)
            token_type_ids = torch.stack(batch['token_type_ids'], dim=1).to(device)
            attention_mask = torch.stack(batch['attention_mask'], dim=1).to(device)

            targets.append(batch['label'].numpy())

            outputs = model(input_ids, token_type_ids, attention_mask)
            preds.append(torch.argmax(outputs, dim=1).to('cpu').numpy())

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        metrics = precision_recall_fscore_support(preds, targets, average='macro')
        precision = metrics[0]
        recall = metrics[1]
        f1 = metrics[2]
        accuracy = accuracy_score(targets, preds)

    return precision, recall, f1, accuracy


def train_loop_wtih_metadata(model, dataloader, device, criterion, optimizer):
    running_loss = 0.0
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        input_ids = torch.stack(batch['input_ids'], dim=1).to(device)
        token_type_ids = torch.stack(batch['token_type_ids'], dim=1).to(device)
        attention_mask = torch.stack(batch['attention_mask'], dim=1).to(device)

        sources = torch.stack(batch['sources_vector'], dim=1).float().to(device)
        subjects = torch.stack(batch['subjects_vector'], dim=1).float().to(device)

        labels = batch['label'].to(device)

        outputs = model(input_ids, token_type_ids, attention_mask, sources, subjects)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

        running_loss += loss.item()
    return running_loss / len(dataloader)


def eval_loop_wtih_metadata(model, dataloader, device):
    preds = []
    targets = []
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader)):
            input_ids = torch.stack(batch['input_ids'], dim=1).to(device)
            token_type_ids = torch.stack(batch['token_type_ids'], dim=1).to(device)
            attention_mask = torch.stack(batch['attention_mask'], dim=1).to(device)

            sources = torch.stack(batch['sources_vector'], dim=1).float().to(device)
            subjects = torch.stack(batch['subjects_vector'], dim=1).float().to(device)

            targets.append(batch['label'].numpy())

            outputs = model(input_ids, token_type_ids, attention_mask, sources, subjects)
            preds.append(torch.argmax(outputs, dim=1).to('cpu').numpy())

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        metrics = precision_recall_fscore_support(preds, targets, average='macro')
        precision = metrics[0]
        recall = metrics[1]
        f1 = metrics[2]
        accuracy = accuracy_score(targets, preds)

    return precision, recall, f1, accuracy