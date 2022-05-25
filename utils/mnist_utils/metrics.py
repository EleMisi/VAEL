import numpy as np
import torch
from torch.nn import functional as F

from utils.graph_semiring import GraphSemiring
from utils.mnist_utils.train import img_log_likelihood


# Reconstructive Ability

def reconstructive_ability(model, test_set, rec_loss='MSE'):
    """Evaluate reconstruction ability of model by computing the reconstruction loss (rec_loss) on test set."""
    model.eval()
    recon_loss = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_set, 1):
            data = torch.as_tensor(images, dtype=torch.float)
            labels = torch.as_tensor(labels, dtype=torch.long)
            data = data[None, ...]
            data = data.permute(1, 0, 2, 3)
            data = data.to(model.device)
            labels = labels.to(model.device)
            recon_batch, mu, logvar, add_prob = model(data, labels)

            if rec_loss == 'BCE':
                recon_loss.append(
                    torch.nn.BCELoss(reduction='mean')(torch.flatten(recon_batch), torch.flatten(data)).cpu().numpy())
            elif rec_loss == 'LAPLACE':
                recon_loss.append(- img_log_likelihood(recon_batch, data).cpu().numpy().mean())
            elif rec_loss == 'MSE':
                recon_loss.append(
                    torch.nn.MSELoss(reduction='mean')(torch.flatten(recon_batch), torch.flatten(data)).cpu().numpy())
            if i == len(test_set):
                break

        return np.mean(recon_loss)


# Discriminative Ability

def acc(pred_labels, true_labels):
    acc = 0
    n = 0
    true_labels = true_labels[:, -2]
    for pred, true in zip(pred_labels, true_labels):
        n += 1
        if pred == true:
            acc += 1
    acc = acc / n
    return acc


def discriminative_ability(model, test_set):
    """Evaluate discriminative ability of model by computing the accuracy on the test set."""

    model.eval()
    class_loss = []
    with torch.no_grad():

        for i, (images, labels) in enumerate(test_set, 1):
            data = torch.as_tensor(images, dtype=torch.float)
            labels = torch.as_tensor(labels, dtype=torch.long)
            data = data[None, ...]
            data = data.permute(1, 0, 2, 3)
            data = data.to(model.device)
            labels = labels.to(model.device)
            _ = model(data, labels)

            queries = list(range(model.mlp.n_facts - 1))
            query_prob = torch.empty(images.shape[0], len(queries))
            for query in queries:
                q_prob, _ = model.problog_inference(model.facts_probs, query=query)
                query_prob[:, query] = q_prob[0]

            pred = torch.argmax(query_prob, dim=1).to(model.device)
            class_loss.append(acc(pred, labels))

            if i == len(test_set):
                break
    return np.mean(class_loss)


# Generative Ability

def generative_ability(model, clf, n_sample):
    model.eval()
    model.semiring = GraphSemiring(n_sample, model.device)
    acc = 0
    for evidence in model.model_dict['evidence'].keys():
        with torch.no_grad():
            # Sample sub-symbolic latent variable
            z = torch.randn(n_sample, model.latent_dim_sym + model.latent_dim_sub)
            z = z.to(model.device)

            # Subsymbolic latent variable
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
            images = model.decode(z_subsym, world_h)

            for image in images:

                # Split image
                img1, img2 = image[0, :, :28], image[0, :, 28:]

                # Classify
                logps1, logps2 = clf(img1.reshape(1, 784)), clf(img2.reshape(1, 784))
                ps1, ps2 = torch.exp(logps1), torch.exp(logps2)
                probab1, probab2 = list(ps1.cpu().numpy()[0]), list(ps2.cpu().numpy()[0])
                digit1, digit2 = probab1.index(max(probab1)), probab2.index(max(probab2))
                if digit1 + digit2 == evidence:
                    acc += 1

    acc = acc / (n_sample * len(model.model_dict['evidence'].keys()))

    return acc
