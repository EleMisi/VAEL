import gc
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from torchvision.utils import make_grid

matplotlib.use('Agg')


def image_reconstruction(model, name, folder, img_suff="", img_dim=(3, 3, 3), test_set=None):
    """Create reconstruction samples of the test set"""
    print("     Reconstruction...")
    folder = os.path.join(folder, str(name), 'images')
    Path(folder).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        model.eval()
        for batch in test_set:
            labels, input_img, output_img = batch['labels'], batch['imgs1'], batch['imgs2']
            data = torch.as_tensor(torch.cat([input_img, output_img], dim=1), dtype=torch.float)
            labels = torch.as_tensor(labels, dtype=torch.long)
            data = data.to(model.device).permute(0,3,1,2)
            labels = labels.to(model.device)

            orig_img, recon_img = model(data, labels)[0:2]

            test_set.reset(shuffle=True)
            break

        batch_images = [
            make_grid([orig.permute(2, 0, 1).detach().cpu(),
                       rec.permute(2, 0, 1).detach().cpu()],
                      nrow=1,
                      padding=1, normalize=True, scale_each=True, pad_value=0).permute(1, 2, 0).numpy() for
            orig, rec in zip(orig_img, recon_img)]
        batch_images1 = np.hstack(batch_images[:len(batch_images) // 2])
        batch_images2 = np.hstack(batch_images[len(batch_images) // 2:])
        batch_images = np.vstack([batch_images1, batch_images2])


    fig = matplotlib.figure.Figure(figsize=(60, 30))
    ax = fig.subplots(1)
    plt.axis('off')
    ax.imshow(batch_images)
    fig.savefig(folder + '/reconstruction_' + img_suff + '.png')

    plt.clf()
    plt.close('all')
    plt.cla()
    gc.collect()


def image_generation(model, name, img_dim=(3, 3, 3), folder='./', img_suff="", n_samples=5, mean=0.0,
                     sigma=1.0):
    """Generate n_samples images and moves"""
    print("     Generation...")
    folder = os.path.join(folder, str(name), 'images')
    Path(folder).mkdir(parents=True, exist_ok=True)

    model.eval()
    batch_labels = []
    n_pos = model.mlp.n_facts // 2
    queries = {'up': (1, 0, 0, 0),
               'down': (0, 1, 0, 0),
               'right': (0, 0, 1, 0),
               'left': (0, 0, 0, 1),
               }
    tuple2query = {v: k for k, v in queries.items()}
    with torch.no_grad():
        for i in range(n_samples):

            # Sample sub-symbolic latent variable
            z = torch.normal(mean, sigma, size=(1, model.latent_dim_sub + model.latent_dim_sym))
            z = z.to(model.device)

            # Split z in two
            z1 = z[:, : model.latent_dim_sym]
            z2 = z[:, model.latent_dim_sym:]

            # Extract prior of initial position facts -> mutually exclusive -> Softmax
            probs_i = torch.nn.Softmax(dim=1)(model.mlp(z1))
            # Initialize uniform probability for final position facts (we use the actions to condition the generation
            # and the final position is deterministic given the initial one, thus the final position probability is uninformative)
            probs_f = torch.ones(size=(1, n_pos)) / n_pos
            # Concatenate facts probability
            probs = torch.cat([probs_i, probs_f.to(model.device)], dim=1)
            # Compute a priori worlds probability P(W)
            worlds_probs = model.compute_worlds_prob_conj(probs)
            # Define queries
            query_prob = torch.empty(1, len(queries))
            # Extract queries probabilities
            for j, query in enumerate(queries):
                # print(q)
                q = torch.as_tensor(queries[query]).to(model.device)
                q = q[None, ...]
                q_prob = model.compute_query_prob(q, worlds_probs)
                query_prob[:, j] = q_prob[0]
            # Sample the action from its prior
            action = torch.nn.functional.gumbel_softmax(logits=torch.log(query_prob), tau=1, hard=True)[0]
            batch_labels.append([tuple2query[tuple(action.tolist())], action])
            # Compute prob for the admissible worlds conditioned to action P(W|c,a)
            cond_worlds_prob = model.compute_conditioned_worlds_prob(probs, action)  # [batch_size, 18]
            # Sample world given P(W|c,e)
            world = torch.nn.functional.gumbel_softmax(logits=torch.log(cond_worlds_prob), tau=1, hard=True)
            # Represent the sampled admissible world in the Herbrand base (truth value for the 18 facts)
            world_h = torch.matmul(world, model.W_adm.type(torch.float))
            # Split the world in the two positions
            world_i, world_f = torch.split(world_h, n_pos, dim=-1)
            # print(world_i,world_f)
            # Image decoding
            image_i = model.decode(z2, world_i)[0].cpu()
            image_f = model.decode(z2, world_f)[0].cpu()

            image = make_grid([image_i, image_f], nrow=1, padding=1, pad_value=0, scale_each=True,
                              normalize=True).permute(1, 2, 0).numpy()

            plt.cla()
            fig = matplotlib.figure.Figure()
            ax = fig.subplots(1)

            ax.imshow(image)

            ax.set_yticks([], [], color='black', fontsize='17', horizontalalignment='right')
            ax.set_xticks([], [], color='black', fontsize='17', horizontalalignment='right')
            ax.set_title(f'Generated action: {tuple2query[tuple(action.tolist())]}\n', size=18)

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)

            fig.savefig(folder + f'/generation_{i}_{img_suff}.png')

            plt.clf()
            plt.cla()
            plt.close('all')
            gc.collect()

    with open(os.path.join(folder, 'generation_labels_' + img_suff + '.txt'), 'w') as f:
        for pred in batch_labels:
            f.write(f"{pred},")


def conditional_image_generation(model, name, test_set, img_dim=(3, 3, 3), folder='./', img_suff="", idx2fact=None):
    """Generate a batch of images from an initial state conditioned on different actions"""
    print("     Conditional generation...")
    folder = os.path.join(folder, str(name), 'images')
    Path(folder).mkdir(parents=True, exist_ok=True)

    model.eval()
    n_pos = model.mlp.n_facts // 2

    batch_list = []
    batch_images = {}
    # Lookup table for admissible moves
    adm_moves = {
        (0, 0): ['up', 'right'],
        (0, 1): ['up', 'down', 'right'],
        (1, 0): ['up', 'right', 'left'],
        (1, 1): ['up', 'down', 'right', 'left'],
        (0, 2): ['down', 'right'],
        (2, 0): ['up', 'left'],
        (1, 2): ['down', 'right', 'left'],
        (2, 1): ['up', 'down', 'left'],
        (2, 2): ['down', 'left']}

    label2pos = {v:k for k,v in test_set.pos2label.items()}
    for batch_idx, batch in enumerate(test_set):
        positions, input_imgs= batch['pos1'],batch['imgs1']
        for pos, img in zip(positions, input_imgs):
            tmp_batch = torch.zeros((5, 3, 100,100))
            with torch.no_grad():
                img = img.type(dtype=torch.float)
                # Store image
                tmp_batch[0] = img.permute(2, 0, 1)
                # Sample latent variable
                mu, logvar = model.encoder(img[None, ...].to(model.device).permute(0,3,1,2))
                z = model.reparametrize(mu, logvar)
                z = z.to(model.device)
                # Split z in two
                z1 = z[:, : model.latent_dim_sym]
                z2 = z[:, model.latent_dim_sym:]
                # Extract prior of initial position facts -> mutually exclusive
                probs_i = torch.nn.Softmax(dim=1)(model.mlp(z1))
                # Initialize uniform probability for final position facts (we use the actions to condition the generation
                # and the final position is deterministic given the initial one, thus the final position probability is uninformative)
                probs_f = torch.ones(size=(1, n_pos)) / n_pos
                # Concatenate facts probability
                probs = torch.cat([probs_i, probs_f.to(model.device)], dim=1)
                # Condition on the admissible moves only

                key = label2pos[pos]
                for move in adm_moves[key]:
                    # Build the evidence from the move
                    e = torch.as_tensor(test_set.move2vec[move], dtype=torch.long).to(model.device)
                    move_idx = int(e.argmax()) + 1
                    # Compute prob for the admissible worlds conditioned to evidence P(W|c,e)
                    cond_worlds_prob = model.compute_conditioned_worlds_prob(probs, e)  # [batch_size, 18]
                    # Sample world given P(W|c,e)
                    # world = torch.nn.functional.gumbel_softmax(logits=torch.log(cond_worlds_prob), tau=1, hard=True)
                    world = cond_worlds_prob.argmax(dim=1)
                    # Debugging
                    if name == 'debug':
                        print(f'-----{move}-----')
                        w = int(world)
                        print("Selected world from {}:\n  {} -> {} (prob = {}, world id={})".format(
                            pos[0].tolist(),
                            idx2fact[int(model.W_adm[w][:9].argmax(0))],
                            idx2fact[int(model.W_adm[w][9:].argmax(0)) + 9], cond_worlds_prob[0, w], w))
                        for w in torch.where(cond_worlds_prob >= float(cond_worlds_prob.max()) - 1e-1)[1]:
                            print("To most probable worlds from {}:\n  {} -> {} (prob = {}, world id={})".format(
                                pos[0].tolist(),
                                idx2fact[int(model.W_adm[w][:9].argmax(0))],
                                idx2fact[int(model.W_adm[w][9:].argmax(0)) + 9], cond_worlds_prob[0, w], w))

                    # world = int(cond_worlds_prob.argmax(0))
                    # Represent the sampled admissible world in the Herbrand base (truth value for the 18 facts)
                    # world_h = torch.matmul(world, model.W_adm.type(torch.float))
                    world_h = model.W_adm.type(torch.float)[world]
                    # Split the world in the two positions
                    world_i, world_f = torch.split(world_h, n_pos, dim=-1)
                    # print(world_i,world_f)
                    # Image decoding
                    image = model.decode(z2, world_f)[0].detach().cpu()
                    # Store image
                    tmp_batch[move_idx] = image

            batch_list.append(
                make_grid(tmp_batch,
                          nrow=5,
                          padding=1, normalize=True, scale_each=True, pad_value=0).permute(1, 2, 0).numpy())

        batch_images[batch_idx] = np.vstack([b for b in batch_list])
        # batch_images = np.vstack([b for b in batch])

        test_set.reset(shuffle=True)
        break

    for batch_idx, idx_images in batch_images.items():
        # fig, ax = plt.subplots(figsize=(60, 20))
        fig = matplotlib.figure.Figure(figsize=(30, 60))
        ax = fig.subplots(1)
        plt.cla()
        ax.imshow(idx_images)

        ax.set_yticks([], [], color='black', fontsize='17', horizontalalignment='right')
        ax.set_xlabel('up, down, right, left', size=18)
        ax.set_xticks([], [], color='black', fontsize='17', horizontalalignment='right')
        """
        plt.xticks([i * 63 for i in range(1, n_worlds+1)], test_set.worlds, color='black', fontsize='18',
                   horizontalalignment='right')
        ax.tick_params(axis=u'both', which=u'both', length=0)
        """
        ax.set_title('Generation with random style vector\n', size=18)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        fig.savefig(folder + '/cond_gen' + img_suff + '.png')

        fig.clear()
        plt.close(fig)
        plt.close('all')


def learning_curve(file_path, name, folder_path='./learning_curve/', save=False,
                   overwrite=False):
    # Load data
    df_train = pd.DataFrame(np.load(os.path.join(file_path, name[0]), allow_pickle=True).tolist())
    df_val = pd.DataFrame(np.load(os.path.join(file_path, name[1]), allow_pickle=True).tolist())
    # df = pd.read_csv(file_path)
    # Prepare folder
    folder_path = folder_path
    if not overwrite:
        if Path(folder_path).is_dir():
            print("Images already existing!")
            save = False
            return 0
    else:
        save = True
    if save:
        Path(folder_path).mkdir(parents=True, exist_ok=True)

    # ELBO (with weighted terms)
    df_train['elbo'].plot(grid=True, figsize=(30, 20), linestyle='dotted', marker='o', color='gray',
                          markerfacecolor='blue', label='train')
    df_val['elbo'].plot(grid=True, figsize=(30, 20), linestyle='dotted', marker='o', color='gray',
                        markerfacecolor='red', label='val')
    plt.title("ELBO", size=20)
    plt.xlabel("Epochs", size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.legend(fontsize=20)
    if save:
        plt.savefig(folder_path + "/ELBO.png")
    else:
        plt.show()

    plt.clf()
    plt.close('all')
    gc.collect()

    # ELBO (with weighted terms e MA)
    df_train['elbo'].rolling(window=50).mean().plot(grid=True, figsize=(30, 20), linestyle='dotted', marker='o',
                                                    color='gray', markerfacecolor='blue', label='train')
    df_val['elbo'].rolling(window=50).mean().plot(grid=True, figsize=(30, 20), linestyle='dotted', marker='o',
                                                  color='gray', markerfacecolor='red', label='val')
    plt.title("ELBO (MA 50)", size=20)
    plt.xlabel("Epochs", size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.legend(fontsize=20)
    if save:
        plt.savefig(folder_path + "/ELBO (MA 50).png")
    else:
        plt.show()

    plt.clf()
    plt.close('all')
    gc.collect()

    losses_list = ['elbo',
                   'recon_loss1',
                   'recon_loss2',
                   'gauss_kl_div1',
                   'gauss_kl_div2',
                   'queryBCE',
                   'labelBCE']

    # Single Terms with weights
    df_train[['recon_loss1', 'gauss_kl_div1', 'recon_loss2', 'gauss_kl_div2', 'queryBCE']].plot(grid=True,
                                                                                                figsize=(30, 20),
                                                                                                linestyle='dotted',
                                                                                                marker='o', alpha=0.6)

    plt.title("Train Single Terms", size=20)
    plt.xlabel("Epochs", size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.legend(fontsize=20)
    if save:
        plt.savefig(folder_path + "/Train_Single_Terms.png")
    else:
        plt.show()

    plt.clf()
    plt.close('all')
    gc.collect()

    df_val[['recon_loss1', 'gauss_kl_div1', 'recon_loss2', 'gauss_kl_div2', 'queryBCE']].plot(grid=True,
                                                                                              figsize=(30, 20),
                                                                                              linestyle='dotted',
                                                                                              marker='o', alpha=0.6)

    plt.title("Val Single Terms", size=20)
    plt.xlabel("Epochs", size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.legend(fontsize=20)
    if save:
        plt.savefig(folder_path + "/Val_Single_Terms.png")
    else:
        plt.show()

    plt.clf()
    plt.close('all')
    gc.collect()

    # Single Terms normalized
    x = df_train.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df2_train = pd.DataFrame(x_scaled, columns=df_train.columns)
    df2_train[['recon_loss1', 'gauss_kl_div1', 'recon_loss2', 'gauss_kl_div2', 'queryBCE']].plot(grid=True,
                                                                                                 figsize=(30, 20),
                                                                                                 linestyle='dotted',
                                                                                                 marker='o', alpha=0.6)
    plt.title("Train Single Terms (NORMALIZED)", size=20)
    plt.xlabel("Epochs", size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.legend(fontsize=20)
    if save:
        plt.savefig(folder_path + "/Train_Single_(NORMALIZED).png")
    else:
        plt.show()

    plt.clf()
    plt.close('all')
    gc.collect()

    x = df_val.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df2_val = pd.DataFrame(x_scaled, columns=df_val.columns)
    df2_val[['recon_loss1', 'gauss_kl_div1', 'recon_loss2', 'gauss_kl_div2', 'queryBCE']].plot(grid=True,
                                                                                               figsize=(30, 20),
                                                                                               linestyle='dotted',
                                                                                               marker='o', alpha=0.6)
    plt.title("Val Single Terms (NORMALIZED)", size=20)
    plt.xlabel("Epochs", size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.legend(fontsize=20)
    if save:
        plt.savefig(folder_path + "/Val_Single_Terms_(NORMALIZED).png")
    else:
        plt.show()

    plt.clf()
    plt.close('all')
    gc.collect()

    # Single Terms normalized with MA
    df2_train[['recon_loss1', 'gauss_kl_div1', 'recon_loss2', 'gauss_kl_div2', 'queryBCE']].rolling(
        window=50).mean().plot(grid=True, figsize=(30, 20),
                               linestyle='dotted', marker='o',
                               alpha=0.6)
    plt.title("Train Single Terms (NORMALIZED MA 50)", size=20)
    plt.xlabel("Epochs", size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.legend(fontsize=20)
    if save:
        plt.savefig(folder_path + "/Train_Single_Terms_(NORMALIZED MA).png")
    else:
        plt.show()

    plt.clf()
    plt.close('all')
    gc.collect()

    df2_val[['recon_loss1', 'gauss_kl_div1', 'recon_loss2', 'gauss_kl_div2', 'queryBCE']].rolling(
        window=50).mean().plot(grid=True, figsize=(30, 20),
                               linestyle='dotted', marker='o',
                               alpha=0.6)
    plt.title("Val Single Terms (NORMALIZED MA 50)", size=20)
    plt.xlabel("Epochs", size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.legend(fontsize=20)
    if save:
        plt.savefig(folder_path + "/Val_Single_Terms_(NORMALIZED MA).png")
    else:
        plt.show()

    plt.clf()
    plt.close('all')
    gc.collect()
