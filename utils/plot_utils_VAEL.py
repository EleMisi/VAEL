import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from torch.nn import functional as F
from torchvision.utils import make_grid

from utils.graph_semiring import GraphSemiring


def image_reconstruction(model, name, folder, img_suff="", img_dim=[28, 56], test_set=None, samples_x_world=320):
    """Create reconstruction samples of the test set"""
    print("     Reconstruction...")
    folder = os.path.join(folder, str(name), 'images')
    Path(folder).mkdir(parents=True, exist_ok=True)

    for k in test_set.world_counter:
        test_set.world_counter[k] = 1

    model.semiring = GraphSemiring(batch_size=samples_x_world, device=model.device)
    n_worlds = len(test_set.worlds)
    batch_images = []
    with torch.no_grad():
        model.eval()
        for j, (images, labels) in enumerate(test_set, 1):
            recon = torch.empty(samples_x_world, *img_dim)
            original = torch.empty(samples_x_world, *img_dim)
            for i, (image, label) in enumerate(zip(images, labels)):
                image = torch.as_tensor(image, dtype=torch.float)
                image = image[None, None, ...]
                label = torch.as_tensor(label, dtype=torch.long)
                label = label[None, ...]
                image = image.permute(1, 0, 2, 3)
                image = image.to(model.device)
                label = label.to(model.device)

                recon_img = model(image, label)[0]
                recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min())
                recon[i] = recon_img

                image = (image - image.min()) / (image.max() - image.min())
                original[i] = image

            img1 = make_grid(torch.Tensor(original[:32]).reshape(-1, 1, 28, 56), 8, 4, pad_value=1).permute(1, 2, 0)
            img2 = make_grid(torch.Tensor(recon[:32]).reshape(-1, 1, 28, 56), 8, 4, pad_value=1).permute(1, 2, 0)
            batch_images.append(np.vstack([img1, img2]))

            if j == n_worlds:
                break

    imgs = [np.hstack(batch_images[i:i + 10]) for i in range(0, 91, 10)]
    img = np.vstack(imgs)

    plt.figure(figsize=(60, 30))
    plt.axis('off')
    plt.imshow(img)
    plt.savefig(folder + '/reconstruction_' + img_suff + '.png')
    plt.close('all')


def image_generation(model, name, folder, img_suff="", n_samples=9, batch_size=32, mean=0.0, sigma=1.0):
    """Generate n_samples images and labels"""
    print("     Generation...")
    folder = os.path.join(folder, str(name), 'images')
    Path(folder).mkdir(parents=True, exist_ok=True)

    model.eval()
    model.semiring = GraphSemiring(batch_size=batch_size, device=model.device)
    batch_images = []
    batch_labels = []

    with torch.no_grad():
        for i in range(n_samples):

            # Sample sub-symbolic latent variable
            z = torch.normal(mean, sigma, size=(1, model.latent_dim_sub + model.latent_dim_sym))
            z = z.to(model.device)

            # Subsymbolical latent variable
            z_subsym = z[:, model.latent_dim_sym:]

            # Extract probability for each digit
            model.facts_probs = model.compute_facts_probability(z[:, :model.latent_dim_sym])

            queries = list(range(model.mlp.n_facts - 1))
            query_prob = torch.empty(1, len(queries))
            for query in queries:
                q_prob, worlds_prob = model.problog_inference(model.facts_probs, query=query)
                query_prob[:, query] = q_prob[0]
            batch_labels.append(torch.argmax(query_prob, dim=1)[0])

            # Sample a world according to P(w) via gumbel softmax
            logits = torch.log(worlds_prob)
            world = F.gumbel_softmax(logits, tau=1, hard=True)

            # Represent the sampled world in the herbrand base
            world_h = model.herbrand(world)

            # Image decoding
            image = model.decode(z_subsym, world_h).detach().cpu().numpy()[0]
            image = (image - image.min()) / (image.max() - image.min())
            batch_images.append(image)

    fig, ax = plt.subplots(figsize=(60, 20))
    plt.imshow(make_grid(torch.Tensor(batch_images), n_samples // 3, 10, pad_value=1).permute(1, 2, 0), )
    plt.yticks([], [], color='black', fontsize='17', horizontalalignment='right')
    plt.xlabel('worlds', size=18)
    plt.xticks([], [], color='black', fontsize='17', horizontalalignment='right')
    plt.title('Generation with random style vector\n', size=18)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.savefig(folder + '/generation_' + img_suff + '.png')
    plt.close('all')

    with open(os.path.join(folder, 'generation_labels_' + img_suff + '.txt'), 'w') as f:
        for pred in batch_labels:
            f.write(f"{pred},")


def conditional_image_generation(model, name, folder, img_suff="", batch_size=32, mean=0.0, sigma=1.0):
    """Generate 10 images for each possible label"""
    print("     Conditional Generation")
    folder = os.path.join(folder, str(name), 'images')
    Path(folder).mkdir(parents=True, exist_ok=True)

    model.eval()
    model.semiring = GraphSemiring(batch_size=batch_size, device=model.device)
    batch = []
    with torch.no_grad():
        for i in range(10):
            for evidence in model.model_dict['evidence'].keys():
                with torch.no_grad():
                    # Sample sub-symbolic latent variable
                    z = torch.normal(mean, sigma, size=(1, model.latent_dim_sym + model.latent_dim_sub))
                    z = z.to(model.device)

                    # Subsymbolical latent variable
                    z_subsym = z[:, model.latent_dim_sym:]

                    # Extract probability for each digit
                    model.facts_probs = model.compute_facts_probability(z[:, :model.latent_dim_sym])

                    # Problog inference to compute worlds probability distributions given the evidence P(w|e)
                    worlds_prob = model.problog_inference_with_evidence(model.facts_probs, evidence)

                    # Sample a world according to P(w) via gumbel softmax
                    logits = torch.log(worlds_prob)
                    world = F.gumbel_softmax(logits, tau=1, hard=True)

                    # Represent the sampled world in the herbrand base
                    world_h = model.herbrand(world)

                    # Image decoding
                    image = model.decode(z_subsym, world_h).detach().cpu().numpy()[0]
                    image = (image - image.min()) / (image.max() - image.min())
                    batch.append(image)

    fig, ax = plt.subplots(figsize=(60, 20))
    plt.imshow(
        make_grid(torch.Tensor(batch), len(model.model_dict['evidence'].keys()), 10, pad_value=1).permute(1, 2, 0), )
    plt.yticks([], [], color='black', fontsize='17', horizontalalignment='right')
    plt.xlabel('worlds', size=18)
    plt.xticks([], [], color='black', fontsize='17', horizontalalignment='right')
    plt.title('Generation with random style vector\n', size=18)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.savefig(folder + '/cond_generation_' + img_suff + '.png')
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
    plt.xlabel("Iterations", size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.legend(fontsize=20)
    if save:
        plt.savefig(folder_path + "/ELBO.png")
    else:
        plt.show()
    plt.close('all')

    # ELBO (with weighted terms e MA)
    df_train['elbo'].rolling(window=50).mean().plot(grid=True, figsize=(30, 20), linestyle='dotted', marker='o',
                                                    color='gray', markerfacecolor='blue', label='train')
    df_val['elbo'].rolling(window=50).mean().plot(grid=True, figsize=(30, 20), linestyle='dotted', marker='o',
                                                  color='gray', markerfacecolor='red', label='val')
    plt.title("ELBO (MA 50)", size=20)
    plt.xlabel("Iterations", size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.legend(fontsize=20)
    if save:
        plt.savefig(folder_path + "/ELBO (MA 50).png")
    else:
        plt.show()
    plt.close('all')

    # Single Terms with weights

    df_train[['recon_loss', 'kl_div', 'queryBCE']].plot(grid=True, figsize=(30, 20), linestyle='dotted',
                                                        marker='o', alpha=0.6)

    plt.title("Train Single Terms", size=20)
    plt.xlabel("Iterations", size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.legend(fontsize=20)
    if save:
        plt.savefig(folder_path + "/Train_Single_Terms.png")
    else:
        plt.show()
    plt.close('all')

    df_val[['recon_loss', 'kl_div', 'queryBCE']].plot(grid=True, figsize=(30, 20), linestyle='dotted',
                                                      marker='o', alpha=0.6)

    plt.title("Val Single Terms", size=20)
    plt.xlabel("Iterations", size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.legend(fontsize=20)
    if save:
        plt.savefig(folder_path + "/Val_Single_Terms.png")
    else:
        plt.show()
    plt.close('all')

    # Single Terms normalized
    x = df_train.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df2_train = pd.DataFrame(x_scaled, columns=df_train.columns)
    df2_train[['recon_loss', 'kl_div', 'queryBCE']].plot(grid=True, figsize=(30, 20), linestyle='dotted',
                                                         marker='o', alpha=0.6)
    plt.title("Train Single Terms (NORMALIZED)", size=20)
    plt.xlabel("Iterations", size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.legend(fontsize=20)
    if save:
        plt.savefig(folder_path + "/Train_Single_Terms_(NORMALIZED).png")
    else:
        plt.show()
    plt.close('all')

    x = df_val.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df2_val = pd.DataFrame(x_scaled, columns=df_val.columns)
    df2_val[['recon_loss', 'kl_div', 'queryBCE']].plot(grid=True, figsize=(30, 20), linestyle='dotted',
                                                       marker='o', alpha=0.6)
    plt.title("Val Single Terms (NORMALIZED)", size=20)
    plt.xlabel("Iterations", size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.legend(fontsize=20)
    if save:
        plt.savefig(folder_path + "/Val_Single_Terms_(NORMALIZED).png")
    else:
        plt.show()
    plt.close('all')

    # Single Terms normalized with MA
    df2_train[['recon_loss', 'kl_div', 'queryBCE']].rolling(window=50).mean().plot(grid=True, figsize=(30, 20),
                                                                                   linestyle='dotted', marker='o',
                                                                                   alpha=0.6)
    plt.title("Train Single Terms (NORMALIZED MA 50)", size=20)
    plt.xlabel("Iterations", size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.legend(fontsize=20)
    if save:
        plt.savefig(folder_path + "/Train_Single_Terms_(NORMALIZED MA).png")
    else:
        plt.show()
    plt.close('all')

    df2_val[['recon_loss', 'kl_div', 'queryBCE']].rolling(window=50).mean().plot(grid=True, figsize=(30, 20),
                                                                                 linestyle='dotted', marker='o',
                                                                                 alpha=0.6)
    plt.title("Val Single Terms (NORMALIZED MA 50)", size=20)
    plt.xlabel("Iterations", size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.legend(fontsize=20)
    if save:
        plt.savefig(folder_path + "/Val_Single_Terms_(NORMALIZED MA).png")
    else:
        plt.show()
    plt.close('all')

    # ReconLoss with MA
    df_train[['recon_loss']].rolling(window=50).mean().plot(grid=True, figsize=(30, 20), linestyle='dotted', marker='o',
                                                            alpha=0.6, markerfacecolor='blue', label='train')
    df_val[['recon_loss']].rolling(window=50).mean().plot(grid=True, figsize=(30, 20), linestyle='dotted', marker='o',
                                                          alpha=0.6, markerfacecolor='red', label='val')
    plt.title("Recon Loss (MA 50)", size=20)
    plt.xlabel("Iterations", size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.legend(fontsize=20)
    if save:
        plt.savefig(folder_path + "/Recon_Loss_(MA).png")
    else:
        plt.show()
    plt.close('all')

    # kl with MA
    df_train[['kl_div']].rolling(window=50).mean().plot(grid=True, figsize=(30, 20), linestyle='dotted', marker='o',
                                                        alpha=0.6, markerfacecolor='blue', label='train')
    df_val[['kl_div']].rolling(window=50).mean().plot(grid=True, figsize=(30, 20), linestyle='dotted', marker='o',
                                                      alpha=0.6, markerfacecolor='red', label='val')
    plt.title("kl_div (MA 50)", size=20)
    plt.xlabel("Iterations", size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.legend(fontsize=20)
    if save:
        plt.savefig(folder_path + "/KL_div_(MA).png")
    else:
        plt.show()
    plt.close('all')

    # ReconLoss with MA
    df_train[['queryBCE']].rolling(window=50).mean().plot(grid=True, figsize=(30, 20), linestyle='dotted', marker='o',
                                                          alpha=0.6, markerfacecolor='blue', label='train')
    df_val[['queryBCE']].rolling(window=50).mean().plot(grid=True, figsize=(30, 20), linestyle='dotted', marker='o',
                                                        alpha=0.6, markerfacecolor='red', label='val')
    plt.title("CrossEntropy (MA 50)", size=20)
    plt.xlabel("Iterations", size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.legend(fontsize=20)
    if save:
        plt.savefig(folder_path + "/QueryCrossEntropy_(MA).png")
    else:
        plt.show()
    plt.close('all')
