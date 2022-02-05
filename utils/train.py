import os
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from utils.EarlyStopping import EarlyStopping
from utils.graph_semiring import GraphSemiring
from utils.printing_utils import prettify_statistics


def loss_function(x_recon, x, mu, logvar, add_prob, model, labels=None, query=True, recon_w=1, kl_w=1, query_w=1,
                  sup_w=1, sup=False, rec_loss='MSE'):
    # Reconstruction loss
    if rec_loss == 'BCE':
        recon_loss = torch.nn.BCELoss(reduction='mean')(torch.flatten(x_recon), torch.flatten(x))
    elif rec_loss == 'LAPLACE':
        recon_loss = - img_log_likelihood(x_recon, x).mean()
    elif rec_loss == 'MSE':
        recon_loss = torch.nn.MSELoss(reduction='mean')(torch.flatten(x_recon), torch.flatten(x))

    # Gaussian KL divergence with standard prior N(0,1) for z_subsym
    gauss_kl_div = - 0.5 * torch.sum(1. + logvar - mu.pow(2) - logvar.exp(), dim=1)
    gauss_kl_div = torch.mean(gauss_kl_div)

    # Cross Entropy on the query
    if query:
        target = torch.ones_like(add_prob)
        query_cross_entropy = torch.nn.BCELoss(reduction='mean')(torch.flatten(add_prob), torch.flatten(target))
    else:
        query_cross_entropy = torch.zeros(size=())

    # Cross Entropy digits supervision
    if sup:
        idxs = labels[labels[:, -1] > -1][:, -1]  # Index(es) of the labelled image in the current batch
        digit1, digit2 = labels[idxs[0]][:2]  # Correct digit in position 1 and 2, each batch has the same images

        pred_digit1 = model.digits_probs[idxs, 0, digit1]
        pred_digit2 = model.digits_probs[idxs, 1, digit2]
        pred = torch.cat([pred_digit1, pred_digit2])
        target = torch.ones_like(pred, dtype=torch.float32)
        label_cross_entropy = torch.nn.BCELoss(reduction='mean')(torch.flatten(pred), torch.flatten(target))

    else:
        label_cross_entropy = torch.zeros(size=())

    # Total loss
    loss = recon_w * recon_loss + kl_w * gauss_kl_div + query_w * query_cross_entropy + sup_w * label_cross_entropy

    return loss, recon_loss, gauss_kl_div, query_cross_entropy, label_cross_entropy


def img_log_likelihood(recon, xs):
    return torch.distributions.Laplace(recon, torch.ones_like(recon)).log_prob(xs).sum(dim=(1, 2, 3))


def train_PLVAE(model, optimizer, n_epochs, train_set, val_set, folder, early_stopping_info=None, run_ID='_',
                query=True, recon_w=1, kl_w=1, query_w=1, sup_w=0,
                rec_loss='MSE', train_batch_size=32, val_batch_size=32):
    Path(os.path.join(folder, run_ID)).mkdir(parents=True, exist_ok=True)
    if sup_w > 0:
        sup_file = os.path.join(folder, run_ID, run_ID + '_sup.csv')
        sup_file = open(sup_file, 'w')
        sup_file.write("digits,supervision_loss\n")

    # Store all epoch info
    train_info = {}
    validation_info = {}

    # initialize the early_stopping object
    early_stopping = EarlyStopping(**early_stopping_info,
                                   folder=os.path.join(folder, run_ID),
                                   verbose=True)

    for epoch in range(1, n_epochs + 1):

        # Training info
        train_epoch_info = {
            'elbo': [],
            'recon_loss': [],
            'kl_div': [],
            'queryBCE': [],
            'labelBCE': []
        }

        # Validation info
        validation_epoch_info = {
            'elbo': [],
            'recon_loss': [],
            'kl_div': [],
            'queryBCE': [],
            'labelBCE': []
        }

        ############
        # Training #
        ############
        model.train()
        train_set.reset_counter()  # Reset data loader internal counter
        model.semiring = GraphSemiring(batch_size=train_batch_size, device=model.device)

        # Loop over batches
        pbar = tqdm(total=len(train_set), position=0, leave=True)
        for batch, (data, labels) in enumerate(train_set, 1):

            sup = False  # No supervised digits in the current batch

            optimizer.zero_grad()

            batch_size = data.shape[0]

            if labels[:, -1].max() > -1:
                sup = True  # Batch with supervised digits
                sup_file.write(f'{labels[0][:2]},')

            data = torch.as_tensor(data, dtype=torch.float)
            labels = torch.as_tensor(labels, dtype=torch.long)
            data = data[None, ...]

            data = data.to(model.device)
            labels = labels.to(model.device)

            # Forward pass
            recon_batch, mu, logvar, add_prob = model(data.permute(1, 0, 2, 3), labels)
            loss, recon_loss, gauss_kl_div, queryBCE, labelBCE = loss_function(recon_batch,
                                                                               data.permute(1, 0, 2, 3),
                                                                               mu,
                                                                               logvar,
                                                                               add_prob,
                                                                               labels=labels,
                                                                               query=query,
                                                                               model=model,
                                                                               recon_w=recon_w,
                                                                               kl_w=kl_w,
                                                                               query_w=query_w,
                                                                               sup_w=sup_w,
                                                                               sup=sup,
                                                                               rec_loss=rec_loss)

            # Registering losses
            train_epoch_info['elbo'].append(loss.data.cpu().detach().numpy())
            train_epoch_info['recon_loss'].append(recon_loss.data.cpu().detach().numpy())
            train_epoch_info['kl_div'].append(gauss_kl_div.data.cpu().detach().numpy())
            train_epoch_info['queryBCE'].append(queryBCE.data.cpu().detach().numpy())
            train_epoch_info['labelBCE'].append(labelBCE.data.cpu().detach().numpy())

            # Backward pass
            loss.backward()

            # Optimization step
            optimizer.step()

            pbar.update(1)

            if batch == len(train_set):
                break

        pbar.close()

        # Compute epoch training losses
        train_epoch_info = {key: np.mean(value) for key, value in train_epoch_info.items()}

        # Store training epoch info
        for key, value in train_epoch_info.items():
            train_info.setdefault(key, []).append(value)

        # Print epoch training losses
        print('\n{}'.format(prettify_statistics(train_epoch_info)))

        ##############
        # Validation #
        ##############

        val_set.reset_counter()
        model.semiring = GraphSemiring(batch_size=val_batch_size, device=model.device)

        model.eval()

        print("Evaluation...")
        pbar = tqdm(total=len(val_set), position=0, leave=True)
        for batch, (data, labels) in enumerate(val_set, 1):

            with torch.no_grad():
                data = torch.as_tensor(data, dtype=torch.float)
                labels = torch.as_tensor(labels, dtype=torch.long)
                data = data[None, ...]
                data = data.to(model.device)
                labels = labels.to(model.device)

                # Forward pass
                recon_batch, mu, logvar, add_prob = model(data.permute(1, 0, 2, 3), labels)
                loss, recon_loss, gauss_kl_div, queryBCE, labelBCE = loss_function(recon_batch,
                                                                                   data.permute(1, 0, 2, 3),
                                                                                   mu,
                                                                                   logvar,
                                                                                   add_prob,
                                                                                   labels=None,
                                                                                   query=query,
                                                                                   recon_w=recon_w,
                                                                                   kl_w=kl_w,
                                                                                   query_w=query_w,
                                                                                   model=model,
                                                                                   sup=False,
                                                                                   rec_loss=rec_loss)

            # Record validation loss
            validation_epoch_info['elbo'].append(loss.data.cpu().detach().numpy())
            validation_epoch_info['recon_loss'].append(recon_loss.data.cpu().detach().numpy())
            validation_epoch_info['kl_div'].append(gauss_kl_div.data.cpu().detach().numpy())
            validation_epoch_info['queryBCE'].append(queryBCE.data.cpu().detach().numpy())
            validation_epoch_info['labelBCE'].append(labelBCE.data.cpu().detach().numpy())

            pbar.update(1)

            if batch == len(val_set):
                break

        pbar.close()

        # Compute epoch validation losses
        validation_epoch_info = {key: np.mean(value) for key, value in validation_epoch_info.items()}

        # Store validation epoch info
        for key, value in validation_epoch_info.items():
            validation_info.setdefault(key, []).append(value)

        # Print epoch validation losses
        print('\n{}'.format(prettify_statistics(validation_epoch_info)))

        # Reset generator
        val_set.reset_counter()

        # Early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        path = early_stopping(validation_epoch_info['elbo'], model, optimizer)

        if early_stopping.early_stop:
            print("Early stopping at epoch {}!".format(epoch))
            # Save learning curves
            np.save(os.path.join(os.path.join(folder, run_ID), 'train_info.npy'), train_info)
            np.save(os.path.join(os.path.join(folder, run_ID), 'validation_info.npy'), validation_info)
            return path, epoch, train_info, validation_info

    path = os.path.join(folder, 'checkpoint.pt')
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, path)
    return path, epoch, train_info, validation_info
