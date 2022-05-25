import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sn
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from torch import nn

from utils.early_stopping import EarlyStopping


def train(model, optimizer, epochs, train_set, val_set, early_stopping_info, device):

    device = torch.device(device)
    early_stopping = EarlyStopping(patience=early_stopping_info['patience'],
                                   delta=early_stopping_info['delta'],
                                   folder=early_stopping_info['folder'],
                                   name=early_stopping_info['name'],
                                   verbose=True)

    for epoch in range(1, epochs + 1):

        model.train()
        for batch_idx, batch in enumerate(train_set, 1):
            # Define labels
            pos1 = batch['pos1']
            num_classes = 9
            labels = torch.nn.functional.one_hot(torch.tensor(pos1), num_classes=num_classes)

            imgs1 = batch['imgs1']

            optimizer.zero_grad()
            data = torch.as_tensor(imgs1, dtype=torch.float)
            labels = torch.as_tensor(labels, dtype=torch.float)
            data = data.to(device)
            labels = labels.to(device)

            preds = model(data.permute(0, 3, 1, 2))
            target = torch.argmax(labels, dim=-1)
            loss = nn.CrossEntropyLoss(reduction='mean')(preds, target)
            loss.backward()

            optimizer.step()

            if train_set.end:
                train_set.reset(shuffle=True)
                break
        print('Train Epoch: {}/{} ({:.0f}%)\tLoss: {:.6f}'.format(
            epoch, epochs,
            100. * epoch / epochs, loss.data))

        model.eval()

        print("\nEvaluation...")
        val_loss = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_set, 1):
                # Define labels
                pos1 = batch['pos1']
                labels = torch.nn.functional.one_hot(torch.tensor(pos1), num_classes=num_classes)

                imgs1 = batch['imgs1']
                data = torch.as_tensor(imgs1, dtype=torch.float)
                labels = torch.as_tensor(labels, dtype=torch.float)
                data = data.to(device)
                labels = labels.to(device)
                # Prediction
                preds = model(data.permute(0, 3, 1, 2))
                target = torch.argmax(labels, dim=-1)
                loss = nn.CrossEntropyLoss(reduction='mean')(preds, target)
                val_loss.append(loss.data)

                if val_set.end:
                    val_set.reset(shuffle=True)
                    break
        loss = torch.mean(torch.Tensor(val_loss))
        path = early_stopping(loss, model, optimizer)

        if early_stopping.early_stop:
            return path

    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, path)

    return path


def test(model, path, test_set, name, mode, folder, device):
    device = torch.device(device)
    # Load model
    last_checkpoint = torch.load(path)
    model.load_state_dict(last_checkpoint['model'])

    model.eval()
    preds = []
    target = []
    unnormalized_preds = []
    metrics = {
        'f1': 'None',
        'conf_matrix': 'None',
        'acc': 'None'
    }
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_set, 1):
            # Define labels
            pos1 = batch['pos1']
            num_classes = 9
            labels = torch.nn.functional.one_hot(torch.tensor(pos1), num_classes=num_classes)

            imgs1 = batch['imgs1']
            data = torch.as_tensor(imgs1, dtype=torch.float)
            labels = torch.as_tensor(labels, dtype=torch.long)
            data = data.to(device)
            labels = labels.to(device)

            output = model(data.permute(0, 3, 1, 2))
            unnormalized_preds.append(output.cpu().numpy())
            preds.append(torch.argmax(output, dim=1).cpu().numpy())
            target.append(torch.argmax(labels, dim=-1).cpu().numpy())

            if test_set.end:
                test_set.reset(shuffle=True)
                break
    unnormalized_preds = torch.cat([torch.Tensor(p).ravel() for p in unnormalized_preds]).reshape(-1, num_classes)
    preds = torch.cat([torch.Tensor(p).ravel() for p in preds])
    target = torch.cat([torch.Tensor(p).ravel().type(torch.long) for p in target])
    metrics['loss'] = nn.CrossEntropyLoss(reduction='mean')(unnormalized_preds, target)
    metrics['f1'] = f1_score(target.ravel(), preds.ravel(), average='micro')
    metrics['acc'] = accuracy_score(target.ravel(), preds.ravel())
    metrics['conf_matrix'] = confusion_matrix(target.ravel(), preds.ravel())

    if name:
        df_cm = pd.DataFrame(metrics['conf_matrix'], index=[i for i in range(num_classes)],
                             columns=[i for i in range(num_classes)])
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)
        Path(folder).mkdir(parents=True, exist_ok=True)
        path = os.path.join(folder, f'conf_matrix_{mode}.png')
        plt.savefig(path)

        np.save(os.path.join(folder, f'metrics_{mode}.npy'), metrics)

    return metrics
