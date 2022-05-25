import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from tqdm import tqdm

from utils.mario_utils.train import img_log_likelihood


def reconstructive_ability(model, test_set, rec_loss='MSE'):
    """
    Evaluate reconstruction ability of model by computing the reconstruction loss (rec_loss) on test set.
    """

    recon_loss = []
    with torch.no_grad():
        model.eval()
        for batch in test_set:
            labels, input_img, output_img = batch['labels'], batch['imgs1'], batch['imgs2']
            data = torch.as_tensor(torch.cat([input_img, output_img], dim=1), dtype=torch.float)
            labels = torch.as_tensor(labels, dtype=torch.long)
            data = data.to(model.device).permute(0,3,1,2)
            labels = labels.to(model.device)

            orig_img1, recon_img1, _,_, orig_img2, recon_img2, _,_,_ = model(data, labels)
            orig_img = torch.cat([orig_img1,orig_img2], dim = 1)
            recon_img = torch.cat([recon_img1, recon_img2], dim=1)
            if rec_loss == 'BCE':
                recon_loss.append(
                    torch.nn.BCELoss(reduction='mean')(torch.flatten(recon_img), torch.flatten(orig_img)).cpu().numpy())
            elif rec_loss == 'LAPLACE':
                recon_loss.append(- img_log_likelihood(recon_img, orig_img).cpu().numpy().mean())
            elif rec_loss == 'MSE':
                recon_loss.append(
                    torch.nn.MSELoss(reduction='mean')(torch.flatten(recon_img), torch.flatten(orig_img)).cpu().numpy())

            # break
            if test_set.end:
                # Reset data generator and exit loop
                test_set.reset(shuffle=True)
                break

    return np.mean(recon_loss)


def discriminative_ability(model, test_set, name=None, folder='./', mode='val'):
    """
    Evaluate discriminative ability of model by computing the accuracy on the test set.
    """
    model.eval()
    queries = {'up': (1, 0, 0, 0),
               'down': (0, 1, 0, 0),
               'right': (0, 0, 1, 0),
               'left': (0, 0, 0, 1),
               }
    class_loss = []
    metrics = {
        'f1': 'None',
        'conf_matrix': 'None',
        'acc': 'None'
    }
    preds = []
    target = []
    with torch.no_grad():

        for batch in test_set:
            labels, input_img, output_img = batch['labels'], batch['imgs1'], batch['imgs2']
            data = torch.as_tensor(torch.cat([input_img, output_img], dim=1), dtype=torch.float)
            labels = torch.as_tensor(labels, dtype=torch.long)
            data = data.to(model.device).permute(0,3,1,2)
            labels = labels.to(model.device)
            # Fit model on data
            _ = model(data, labels)

            # Extract queries probabilities
            query_prob = torch.empty(model.worlds_probs.shape[0], len(queries))
            for j, query in enumerate(queries):
                # print(q)
                q = torch.as_tensor(queries[query]).to(model.device)
                q = q.expand(query_prob.shape)
                q_prob = model.compute_query_prob(q, model.worlds_probs)
                query_prob[:, j] = q_prob

            preds.append(torch.argmax(query_prob, dim=1).to(model.device).cpu().numpy())
            target.append(torch.argmax(labels, dim=1).to(model.device).cpu().numpy())

            """
            facts = problog_model.FACTS_BASE_MODEL
            idx2fact = {i: e for i, e in
                        enumerate(facts.replace('\n{}::', ' ').replace('\n', ' ').replace(';', ' ').split())}
            fact2idx = {e: i for i, e in idx2fact.items()}
            res = model.W_all[model.worlds_probs.argmax(dim=1)]
            res = np.array(res)
            df_cm = pd.DataFrame(res, index=[target[i] for i in range(len(test_set))],
                                 columns=[idx2fact[i] for i in range(18)])
            plt.figure(figsize=(10, 7))
            sn.heatmap(df_cm, annot=True)
            """
            #break
            if test_set.end:
                # Reset data generator and exit loop
                test_set.reset(shuffle=True)
                break
    try:
        preds = torch.cat([torch.Tensor(p).ravel() for p in preds]).ravel()
        target = torch.cat([torch.Tensor(p).ravel() for p in target]).ravel()
    except:
        print()
    metrics['f1'] = f1_score(target, preds, average='micro')
    metrics['acc'] = accuracy_score(target, preds)
    metrics['conf_matrix'] = confusion_matrix(target, preds)

    if name:
        df_cm = pd.DataFrame(metrics['conf_matrix'], index=[i for i in ['up', 'down', 'right', 'left']],
                             columns=[i for i in ['up', 'down', 'right', 'left']])
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)
        folder = os.path.join(folder, str(name), 'images')
        Path(folder).mkdir(parents=True, exist_ok=True)
        path = os.path.join(folder, f'conf_matrix_{mode}.png')
        plt.savefig(path)
        plt.close('all')

    return metrics['f1'], metrics['acc'], metrics


def generative_ability(model, clf, test_set=None, label2pos=None, n_sample = 3000):
    """
    Evaluate VAEL generative ability by relying on a pre-trained classifier.
    Once provided the initial state (img1) and the move, VAEL is required to conditionally generate the image representing
    the state resulting from performing the given move on the initial state.
    """
    model.eval()
    acc = 0

    if test_set is not None:
        n_sample = 0
        # Iterate over test set samples, one by one
        with torch.no_grad():
            for batch in test_set:
                pos2, labels, input_imgs = batch['pos2'], batch['labels'], batch['imgs1']
                for i in range(len(labels)):
                    input_img = torch.as_tensor(input_imgs[i], dtype=torch.float)
                    evidence = torch.as_tensor(labels[i], dtype=torch.long)
                    input_img = input_img.to(model.device)
                    evidence = evidence.to(model.device)

                    # Image encoding
                    mu1, logvar1 = model.encoder(input_img[None, ...].permute(0,3,1,2))
                    # Sample z
                    z1 = model.reparametrize(mu1, logvar1)
                    # Sub-symbolical component
                    z_subsym1 = z1[:, model.latent_dim_sym:]
                    # Extract prior probabilities for initial position
                    priors1 = torch.nn.Softmax(dim=1)(model.mlp(z1[:, :model.latent_dim_sym]))
                    # Initialize uniform distribution for final position
                    priors2 = torch.ones_like(priors1) * (1 / (model.mlp.n_facts // 2))
                    # P(W)
                    facts_priors = torch.cat([priors1, priors2], dim=1)
                    # Compute conditional probability for the worlds given the move
                    cond_prob = model.compute_conditioned_worlds_prob(facts_priors, evidence)
                    # Sample the world
                    world = cond_prob.argmax(dim=1)
                    # Represent the sampled admissible world in the Herbrand base (truth value for the 18 facts)
                    world_h = model.W_adm.type(torch.float)[world]
                    # Split the world in the two
                    _, world2 = torch.split(world_h, model.mlp.n_facts // 2, dim=-1)
                    # Generate the corresponding image
                    output_gen = model.decode(z_subsym1, world2)[0]
                    # Classify (output_gen)
                    data = torch.as_tensor(output_gen, dtype=torch.float)
                    data = data.to(model.device)[None, ...]
                    clf_output = clf(data)
                    clf_pos = int(torch.argmax(clf_output))
                    target_pos = pos2[i]
                    # print(clf_move.data, move.data)
                    if clf_pos == target_pos:
                        acc += 1
                    n_sample += 1
                # break
                if test_set.end:
                    # Reset data generator and exit loop
                    test_set.reset(shuffle=True)
                    break

    else:
        # Iterate over test set samples, one by one
        n_pos = 9
        moves = {'up': (1, 0, 0, 0),
                   'down': (0, 1, 0, 0),
                   'right': (0, 0, 1, 0),
                   'left': (0, 0, 0, 1),
                   }
        tuple2moves = {v: k for k, v in moves.items()}
        with torch.no_grad():
            for i in range(n_sample):

                # Sample sub-symbolic latent variable
                z = torch.normal(0.0, 1.0, size=(1, model.latent_dim_sub + model.latent_dim_sym))
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
                query_prob = torch.empty(1, len(moves))
                # Extract queries probabilities
                for j, query in enumerate(moves):
                    # print(q)
                    q = torch.as_tensor(moves[query]).to(model.device)
                    q = q[None, ...]
                    q_prob = model.compute_query_prob(q, worlds_probs)
                    query_prob[:, j] = q_prob[0]
                # Sample the action from its prior
                action = torch.nn.functional.gumbel_softmax(logits=torch.log(query_prob), tau=1, hard=True)[0]
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
                # Classify agent position
                data_i = torch.as_tensor(image_i, dtype=torch.float)
                data_i = data_i.to(model.device)[None, ...]
                clf_output_i = clf(data_i)
                clf_pos_1 = int(torch.argmax(clf_output_i))

                data_f = torch.as_tensor(image_f, dtype=torch.float)
                data_f = data_f.to(model.device)[None, ...]
                clf_output_f = clf(data_f)
                clf_pos_2 = int(torch.argmax(clf_output_f))
                # From positions to move
                x0, y0 = label2pos[clf_pos_1]
                x1, y1 = label2pos[clf_pos_2]
                classified_move = pos2move(x0, y0, x1, y1)
                gen_move = tuple2moves[tuple(action.tolist())]
                if classified_move == gen_move:
                    acc += 1

    acc = acc / n_sample

    return acc


def  pos2move(x0,y0,x1,y1):
    if x0 < x1 and y0==y1:
        return "right"
    elif x0 > x1 and  y0==y1:
        return "left"
    elif y0 < y1 and x0==x1:
        return "up"
    elif y0 > y1 and x0==x1:
        return "down"
    else:
        return "not allowed"