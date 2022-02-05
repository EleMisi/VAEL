import torch
from problog.logic import Constant, Term
from torch import nn
from torch.nn import functional as F

from utils.graph_semiring import GraphSemiring


class VAELModel(nn.Module):

    def __init__(self, encoder, decoder, mlp, latent_dims, model_dict, dropout=None, is_train=True, device='cpu'):

        super(VAELModel, self).__init__()

        self.encoder = encoder
        self.mlp = mlp
        self.decoder = decoder
        self.dropout = dropout
        self.latent_dim_sym, self.latent_dim_sub = latent_dims
        self.model_dict = model_dict  # Dictionary of pre-compiled ProbLog models
        self.is_train = is_train
        self.device = device

        # Herbrand base
        eye = torch.eye(self.mlp.n_digits)
        self.herbrand_base = torch.cat(
            [torch.cat((eye[i].expand(self.mlp.n_digits, self.mlp.n_digits), eye), dim=1) for i in
             range(self.mlp.n_digits)], dim=0)

        self.herbrand_base = self.herbrand_base.to(self.device)

        # Weights dictionary for problog inference
        self.weights_dict = {}
        for i in range(1, 3):
            for j in range(self.mlp.n_digits):
                key = 'p_' + str(i) + str(j)
                self.weights_dict[Term(key)] = "NOT DEFINED"

    def decode(self, z, w):
        """
        z: (bs, latent_dim)
        w: (bs, herbrand_base)
        """
        dec_input = torch.cat([z, w], 1)
        image = self.decoder(dec_input)
        return image

    def forward(self, x, labels):

        # Image encoding
        mu, logvar = self.encoder(x)
        # Sample z
        z = self.reparametrize(mu, logvar)

        # Sub-symbolical component
        self.z_subsym = z[:, self.latent_dim_sym:]
        # Dropout on sub-symbolical variable
        if self.dropout:
            self.z_subsym = nn.Dropout(p=self.dropout)(self.z_subsym)

        # Extract probs for each digit
        prob_digit1, prob_digit2 = self.mlp(z[:, : self.latent_dim_sym])
        # Sotfmax on digits_probs (the 10 digits values are mutually exclusive)
        prob_digit1 = nn.Softmax(dim=1)(prob_digit1)
        prob_digit2 = nn.Softmax(dim=1)(prob_digit2)
        # Clamp digits_probs to avoid ProbLog underflow
        eps = 1e-5
        prob_digit1 = prob_digit1 + eps
        with torch.no_grad():
            Z1 = torch.sum(prob_digit1, dim=-1, keepdim=True)
        prob_digit1 = prob_digit1 / Z1  # Normalization
        prob_digit2 = prob_digit2 + eps
        with torch.no_grad():
            Z2 = torch.sum(prob_digit2, dim=-1, keepdim=True)
        prob_digit2 = prob_digit2 / Z2  # Normalization

        # Symbolical component
        self.digits_probs = torch.stack([prob_digit1, prob_digit2], dim=1)

        # Problog inference to compute worlds and query probability distributions
        query = int(labels[0][-2])  # Query of the current batch
        self.query_prob, self.worlds_prob = self.problog_inference(self.digits_probs, query=query)

        # Sample a world according to P(w) via gumbel softmax
        logits = torch.log(self.worlds_prob)
        self.world = F.gumbel_softmax(logits, tau=1, hard=True)

        # Represent the sampled world in the herbrand base
        world_h = self.herbrand(self.world)
        # Image decoding
        image = self.decode(self.z_subsym, world_h)

        return image, mu, logvar, self.query_prob

    def reparametrize(self, mu, logvar):
        """Riparametrization trick to sample from a Gaussian."""
        if self.is_train:
            eps = torch.randn(mu.shape[0], mu.shape[1])
            eps = eps.to(self.device)
            z = mu + torch.exp(logvar / 2) * eps
            return z
        else:
            return mu

    def mlp_accuracy(self, pred_labels, true_labels, N=10000):
        perfect_match = 0
        partial_match = 0

        for pred, true in zip(pred_labels, true_labels):
            # print(pred, true)
            if (pred == true).all():
                perfect_match += 1
            elif pred[0] == true[0] or pred[1] == true[1]:
                partial_match += 1

        acc = (perfect_match + 0.5 * partial_match) / N
        return perfect_match, partial_match, acc

    def herbrand(self, world):
        """Herbrand representation of the current world."""
        return torch.matmul(world, self.herbrand_base)

    def problog_inference(self, digits_probs, query):
        """
        Perform ProbLog inference to retrieve the worlds probability distribution P(w) and the desired query probability.
        """

        n_digits = self.mlp.n_digits

        # Update weights of graph semiring
        # term is 'p_PositionDigit', we use it to index priors
        # digits_probs: (bs, 2, n_digits)
        for term in self.weights_dict:
            str_term = str(term)
            i = int(str_term[-2]) - 1
            j = int(str_term[-1])
            self.weights_dict[term] = digits_probs[:, i, j]

        self.semiring = GraphSemiring(digits_probs.shape[0], self.device)
        self.semiring.set_weights(self.weights_dict)

        # Select pre-compiled ProbLog model corresponding to the query
        sdd = self.model_dict['query'][query]

        # Evaluate model
        res = sdd.evaluate(semiring=self.semiring)

        # Extract query probability
        res_keys = list(res.keys())
        query_prob = res[res_keys[-1]][..., None]

        # Extract ordered worlds probability P(w)
        digits = Term('digits')
        probabilities = []
        for j in range(n_digits):
            for k in range(n_digits):
                term = digits(Constant(j), Constant(k))
                probabilities.append(res[term])
        probabilities = [t[:digits_probs.shape[0]] for t in probabilities]
        # Clamp probabilities to avoid nan
        probabilities = torch.stack(probabilities, dim=1)
        eps = 1e-7
        probabilities = probabilities + eps
        with torch.no_grad():
            P = probabilities.sum()
        self.worlds_prob = probabilities / P

        return query_prob, self.worlds_prob

    def problog_inference_with_evidence(self, digits_probs, evidence):
        """
        Perform ProbLog inference to retrieve the worlds probability distribution P(w) given the evidence.
        """

        n_digits = self.mlp.n_digits

        # Update weights of graph semiring
        # term is 'p_PositionDigit', we use it to index priors
        # digits_probs: (bs, 2, n_digits)
        for term in self.weights_dict:
            str_term = str(term)
            i = int(str_term[-2]) - 1
            j = int(str_term[-1])
            self.weights_dict[term] = digits_probs[:, i, j]
        self.semiring = GraphSemiring(digits_probs.shape[0], self.device)
        self.semiring.set_weights(self.weights_dict)

        # Select pre-compiled ProbLog model corresponding to the evidence
        sdd = self.model_dict['evidence'][evidence]

        # Evaluate model
        res = sdd.evaluate(semiring=self.semiring)

        # Extract ordered worlds probability P(w)
        digits = Term('digits')
        probabilities = []
        for j in range(n_digits):
            for k in range(n_digits):
                term = digits(Constant(j), Constant(k))
                probabilities.append(res[term])
        probabilities = [t[:digits_probs.shape[0]] for t in probabilities]
        # Clamp probabilities to avoid nan
        probabilities = torch.stack(probabilities, dim=1)
        eps = 1e-7
        probabilities = probabilities + eps
        with torch.no_grad():
            P = probabilities.sum()
        worlds_prob = probabilities / P

        return worlds_prob


class MNISTPairsVAELModel(VAELModel):

    def __init__(self, encoder, decoder, mlp, latent_dims, model_dict, w_q, dropout=None, is_train=True, device=False):
        super(MNISTPairsVAELModel, self).__init__(encoder=encoder, decoder=decoder, mlp=mlp, latent_dims=latent_dims,
                                                  model_dict=model_dict, dropout=dropout, is_train=is_train,
                                                  device=device)

        self.w_q = w_q  # Worlds-queries matrix

    def compute_query(self, query, worlds_prob):
        """Compute query probability given the worlds probability P(w)."""
        # Select the column of w_q matrix corresponding to the current query
        w_q = self.w_q[:, query]
        # Compute query probability by summing the probability of all the worlds where the query is true
        query_prob = torch.sum(w_q * worlds_prob, dim=1, keepdim=True)
        return query_prob

    def problog_inference(self, digits_probs, query):
        """
        Perform ProbLog inference to retrieve the worlds probability distribution P(w) and the desired query probability.
        """

        # Extract first and second digit probability
        prob_digit1, prob_digit2 = digits_probs[:, 0], digits_probs[:, 1]
        # Compute worlds probability P(w) (the two digits values are independent)
        Z_1 = prob_digit1[..., None]
        Z_2 = prob_digit2[:, None, :]
        probs = Z_1.multiply(Z_2)
        self.worlds_prob = probs.reshape(-1, self.mlp.n_digits * self.mlp.n_digits)
        # Compute query probability P(q)
        query_prob = self.compute_query(query, self.worlds_prob)

        return query_prob, self.worlds_prob
