import os
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from utils.early_stopping import EarlyStopping


def loss_function(x1, x_recon1, mu1, logvar1, x2, x_recon2, mu2, logvar2, query_prob, recon_w=1, kl_w=1, query_w=1,
                  sup_w=1, sup=False, rec_loss='MSE'):
    # Reconstruction loss of the two images
    try:
        if rec_loss == 'BCE':
            recon_loss1 = torch.nn.BCELoss(reduction='mean')(torch.flatten(x_recon1), torch.flatten(x1))
            recon_loss2 = torch.nn.BCELoss(reduction='mean')(torch.flatten(x_recon2), torch.flatten(x2))
        elif rec_loss == 'LAPLACE':
                recon_loss1 = - img_log_likelihood(x_recon1, x1).mean()
                recon_loss2 = - img_log_likelihood(x_recon2, x2).mean()
        elif rec_loss == 'MSE':
            recon_loss1 = torch.nn.MSELoss(reduction='mean')(torch.flatten(x_recon1), torch.flatten(x1))
            recon_loss2 = torch.nn.MSELoss(reduction='mean')(torch.flatten(x_recon2), torch.flatten(x2))
        else:
            raise ValueError("Unknown reconstruction loss! Valid input are 'BCE', 'LAPLACE', or 'MSE'")
    except:
        print("Error")

    # Gaussian KL divergence with standard prior N(0,1) of the two z_subsym
    gauss_kl_div1 = - 0.5 * torch.sum(1. + logvar1 - mu1.pow(2) - logvar1.exp(), dim=1)
    gauss_kl_div1 = torch.mean(gauss_kl_div1)
    gauss_kl_div2 = - 0.5 * torch.sum(1. + logvar2 - mu2.pow(2) - logvar2.exp(), dim=1)
    gauss_kl_div2 = torch.mean(gauss_kl_div2)

    # Cross Entropy on the query
    target = torch.ones_like(query_prob)
    queryBCE = torch.nn.BCELoss(reduction='mean')(torch.flatten(query_prob), torch.flatten(target))
    # print(query_prob.max())

    # Cross Entropy digits supervision
    if sup:
        # TODO: supervision
        labelBCE = torch.zeros(size=())

    else:
        labelBCE = torch.zeros(size=())

    # ELBO
    elbo = (recon_w * (recon_loss1 + recon_loss2) +
            kl_w * (gauss_kl_div1 + gauss_kl_div2) +
            query_w * queryBCE +
            sup_w * labelBCE)

    true_elbo = recon_loss1 + recon_loss2 + gauss_kl_div1 + gauss_kl_div2 + queryBCE + labelBCE

    losses = {'elbo': elbo,
              'true_elbo': true_elbo.detach(),
              'recon_loss1': recon_loss1,
              'recon_loss2': recon_loss2,
              'gauss_kl_div1': gauss_kl_div1,
              'gauss_kl_div2': gauss_kl_div2,
              'queryBCE': queryBCE,
              'labelBCE': labelBCE}

    return losses


def img_log_likelihood(recon, xs):
    return torch.distributions.Laplace(recon, torch.ones_like(recon)).log_prob(xs).sum(dim=(1, 2, 3))

def train(model, optimizer, n_epochs, train_set, val_set, folder, early_stopping_info=None, run_ID='_',
          recon_w=1, kl_w=1, query_w=1, sup_w=0, rec_loss='MSE'):
    Path(os.path.join(folder, run_ID)).mkdir(parents=True, exist_ok=True)
    if sup_w > 0:
        sup_file = os.path.join(folder, run_ID, run_ID + '_sup.csv')
        sup_file = open(sup_file, 'w')
        sup_file.write("labels,supervision_loss\n")

    # Store all epoch info
    train_info = {}
    validation_info = {}

    # Initialize the early_stopping object
    early_stopping = EarlyStopping(patience=early_stopping_info['patience'],
                                   delta=early_stopping_info['delta'],
                                   folder=os.path.join(folder, run_ID),
                                   verbose=False)
    # List of losses
    losses_list = ['elbo',
                   'true_elbo',
                   'recon_loss1',
                   'recon_loss2',
                   'gauss_kl_div1',
                   'gauss_kl_div2',
                   'queryBCE',
                   'labelBCE']

    pbar = tqdm(total=n_epochs, position=0, leave=True)
    for epoch in range(1, n_epochs + 1):
        # view_used_mem()

        # Training and validation info
        train_epoch_info = {loss: [] for loss in losses_list}
        validation_epoch_info = {loss: [] for loss in losses_list}

        ############
        # Training #
        ############
        model.train()

        # Debug
        # if optimizer_lagrangian is not None:
        #     print(f'Multipliers: {[recon_w, kl_w, query_w]}')

        # Loop over batches
        # pbar = tqdm(total=len(train_set), position=0, leave=True)
        for batch_idx, batch in enumerate(train_set):
            labels = batch['labels']
            input_img = batch['imgs1']
            output_img = batch['imgs2']
            # Reset gradient
            optimizer.zero_grad()

            data = torch.as_tensor(torch.cat([input_img, output_img], dim=1), dtype=torch.float)
            labels = torch.as_tensor(labels, dtype=torch.long)

            data = data.to(model.device).permute(0,3,1,2)
            labels = labels.to(model.device)

            # Forward pass
            x1, x_recon1, mu1, logvar1, x2, x_recon2, mu2, logvar2, query_prob = model(data, labels)

            losses = loss_function(x1,
                                   x_recon1,
                                   mu1,
                                   logvar1,
                                   x2,
                                   x_recon2,
                                   mu2,
                                   logvar2,
                                   query_prob,
                                   recon_w=recon_w,
                                   kl_w=kl_w,
                                   query_w=query_w,
                                   sup_w=sup_w,
                                   sup=False,
                                   rec_loss=rec_loss)

            # Registering losses
            for loss_name, loss in losses.items():
                train_epoch_info.setdefault(loss_name, []).append(loss.cpu().detach().numpy())

            # Backward pass
            losses['elbo'].backward()

            # Optimization step
            optimizer.step()

            batch_idx += 1
            #break
            if train_set.end:
                # Reset data generator and exit loop
                train_set.reset(shuffle=True)
                break

        # Compute epoch training losses
        train_epoch_info = {key: np.mean(value) for key, value in train_epoch_info.items()}

        # Store training epoch info
        for key, value in train_epoch_info.items():
            train_info.setdefault(key, []).append(value)


        ##############
        # Validation #
        ##############

        model.eval()

        # print("\nEvaluation...")
        # pbar = tqdm(total=len(val_set), position=0, leave=True)
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_set):

                labels = batch['labels']
                input_img = batch['imgs1']
                output_img = batch['imgs2']

                data = torch.as_tensor(torch.cat([input_img, output_img], dim=1), dtype=torch.float)
                labels = torch.as_tensor(labels, dtype=torch.long)

                data = data.to(model.device)
                labels = labels.to(model.device)

                # Forward pass
                x1, x_recon1, mu1, logvar1, x2, x_recon2, mu2, logvar2, query_prob = model(data.permute(0,3,1,2), labels)
                losses = loss_function(x1,
                                       x_recon1,
                                       mu1,
                                       logvar1,
                                       x2,
                                       x_recon2,
                                       mu2,
                                       logvar2,
                                       query_prob,
                                       recon_w=recon_w,
                                       kl_w=kl_w,
                                       query_w=query_w,
                                       sup_w=sup_w,
                                       sup=False,
                                       rec_loss=rec_loss)

                # Registering losses
                for loss_name in losses_list:
                    validation_epoch_info[loss_name].append(losses[loss_name].data.cpu().detach().numpy())

                batch_idx += 1
                #break
                if val_set.end:
                    # Reset data generator and exit loop
                    val_set.reset(shuffle=True)
                    break


        # Compute epoch validation losses
        validation_epoch_info = {key: np.mean(value) for key, value in validation_epoch_info.items()}

        # Store validation epoch info
        for key, value in validation_epoch_info.items():
            validation_info.setdefault(key, []).append(value)

        # Print epoch validation losses
        # print('\n{}'.format(prettify_statistics(validation_epoch_info)))

        # Early_stopping needs the validation elbo to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        path = early_stopping(validation_epoch_info[early_stopping_info['loss']], model, optimizer)

        pbar.update(1)

        if early_stopping.early_stop:
            print("Early stopping at epoch {}!".format(epoch))
            # Save learning curves
            np.save(os.path.join(os.path.join(folder, run_ID), 'train_info.npy'), train_info)
            np.save(os.path.join(os.path.join(folder, run_ID), 'validation_info.npy'), validation_info)
            pbar.close()
            return path, epoch, train_info, validation_info

    pbar.close()

    path = os.path.join(folder, run_ID, 'checkpoint.pt')
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, path)

    return path, epoch, train_info, validation_info
