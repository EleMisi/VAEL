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
        self.herbrand_base = self.define_herbrand_base(self.mlp.n_facts).to(self.device)

        # Weights dictionary for problog inference
        self.weights_dict = self.build_weights_dictionary(self.mlp.n_facts)

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

        # Extract probability for each fact from the symbolical component of the latent space
        self.facts_probs = self.compute_facts_probability(z[:, :self.latent_dim_sym])

        # Problog inference to compute worlds and query probability distributions
        self.query_prob, self.worlds_prob = self.problog_inference(self.facts_probs, labels=labels)

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

    def herbrand(self, world):
        """Herbrand representation of the given world(s)"""
        return torch.matmul(world, self.herbrand_base)

    def define_herbrand_base(self, n_facts):
        """Defines the herbrand base to encode ProbLog worlds"""
        pass

    def build_weights_dictionary(self, n_facts):
        """Returns the weights dictionary used during ProbLog inference to update the graph semiring."""
        pass

    def compute_facts_probability(self, z):
        """Computes the probability for each ProbLog fact given the latent vector z"""
        pass

    def problog_inference(self, facts_probs, labels, query):
        """
        Performs ProbLog inference to retrieve the worlds probability distribution P(w) and the desired query probability.
        """
        # Update weights of graph semiring with the facts probability
        self.update_semiring_weights(facts_probs)

        # Select pre-compiled ProbLog model corresponding to the query
        sdd = self.model_dict['query'][labels]

        # Evaluate model
        res = sdd.evaluate(semiring=self.semiring)

        # Extract query probability
        query_prob = self.extract_query_probability(res)

        # Extract worlds probability P(w)
        self.worlds_prob = self.extract_worlds_probability(res)

        return query_prob, self.worlds_prob

    def problog_inference_with_evidence(self, facts_probs, evidence):
        """
        Performs ProbLog inference to retrieve the worlds probability distribution P(w) given the evidence.
        """
        # Update weights of graph semiring with the facts probability
        self.update_semiring_weights(facts_probs)

        # Select pre-compiled ProbLog model corresponding to the evidence
        sdd = self.model_dict['evidence'][evidence]

        # Evaluate model
        res = sdd.evaluate(semiring=self.semiring)

        # Extract worlds probability P(w)
        worlds_prob = self.extract_worlds_probability(res)

        return worlds_prob

    def update_semiring_weights(self, facts_probs):
        """Updates weights of graph semiring with the facts probability"""
        pass

    def extract_worlds_probability(self, res):
        """Extracts P(q) contained in the dictionary 'res' resulting from ProbLog model evaluation."""
        pass

    def extract_query_probability(self, res):
        """Extracts P(w) contained in the dictionary 'res' resulting from ProbLog model evaluation."""
        pass


class MNISTPairsVAELModel(VAELModel):

    def __init__(self, encoder, decoder, mlp, latent_dims, model_dict, w_q, dropout=None, is_train=True, device=False):
        super(MNISTPairsVAELModel, self).__init__(encoder=encoder, decoder=decoder, mlp=mlp, latent_dims=latent_dims,
                                                  model_dict=model_dict, dropout=dropout, is_train=is_train,
                                                  device=device)

        self.w_q = w_q  # Worlds-queries matrix

    def compute_facts_probability(self, z):
        """Computes the probability for each ProbLog fact given the latent vector z"""
        # Extract probs for each digit
        prob_digit1, prob_digit2 = self.mlp(z)
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

        return torch.stack([prob_digit1, prob_digit2], dim=1)

    def define_herbrand_base(self, n_facts):
        """Defines the herbrand base to encode ProbLog worlds"""
        n_digits = n_facts // 2
        eye = torch.eye(n_digits)
        herbrand_base = torch.cat(
            [torch.cat((eye[i].expand(n_digits, n_digits), eye), dim=1) for i in
             range(n_digits)], dim=0)

        return herbrand_base

    def build_weights_dictionary(self, n_facts):
        """Returns the weights dictionary used during ProbLog inference to update the graph semiring."""
        n_digits = n_facts // 2
        weights_dict = {}
        for i in range(1, 3):
            for j in range(n_digits):
                key = 'p_' + str(i) + str(j)
                weights_dict[Term(key)] = "NOT DEFINED"

        return weights_dict

    def compute_query(self, query, worlds_prob):
        """Computes query probability given the worlds probability P(w)."""
        # Select the column of w_q matrix corresponding to the current query
        w_q = self.w_q[:, query]
        # Compute query probability by summing the probability of all the worlds where the query is true
        query_prob = torch.sum(w_q * worlds_prob, dim=1, keepdim=True)
        return query_prob

    def problog_inference(self, facts_probs, labels=None, query=None):
        """
        Performs ProbLog inference to retrieve the worlds probability distribution P(w) and the desired query probability.
        """
        if query == None:
            query = int(labels[0][-2])  # Query of the current batch
        n_digits = self.mlp.n_facts // 2
        # Extract first and second digit probability
        prob_digit1, prob_digit2 = facts_probs[:, 0], facts_probs[:, 1]
        # Compute worlds probability P(w) (the two digits values are independent)
        Z_1 = prob_digit1[..., None]
        Z_2 = prob_digit2[:, None, :]
        probs = Z_1.multiply(Z_2)
        self.worlds_prob = probs.reshape(-1, n_digits * n_digits)
        # Compute query probability P(q)
        query_prob = self.compute_query(query, self.worlds_prob)

        return query_prob, self.worlds_prob

    def update_semiring_weights(self, facts_probs):
        """
        Updates weights of graph semiring with the facts probability.
        Each term probability is indicated as 'p_PositionDigit', we use it to index the priors contained in facts_probs.

        Args:
            facts_probs (bs, 2, n_facts)
        """

        for term in self.weights_dict:
            str_term = str(term)
            i = int(str_term[-2]) - 1
            j = int(str_term[-1])
            self.weights_dict[term] = facts_probs[:, i, j]

        self.semiring = GraphSemiring(facts_probs.shape[0], self.device)
        self.semiring.set_weights(self.weights_dict)

    def extract_query_probability(self, res):
        """Extracts P(q) contained in the dictionary 'res' resulting from ProbLog model evaluation."""
        res_keys = list(res.keys())
        return res[res_keys[-1]][..., None]

    def extract_worlds_probability(self, res):
        """Extracts P(w) contained in the dictionary 'res' resulting from ProbLog model evaluation."""
        n_digits = self.mlp.n_facts // 2
        digits = Term('digits')
        probabilities = []
        for j in range(n_digits):
            for k in range(n_digits):
                term = digits(Constant(j), Constant(k))
                probabilities.append(res[term])
        # Clamp probabilities to avoid nan
        probabilities = torch.stack(probabilities, dim=1)
        eps = 1e-7
        probabilities = probabilities + eps
        with torch.no_grad():
            P = probabilities.sum()

        return probabilities / P


class MarioVAELModel(VAELModel):

    def __init__(self, encoder, decoder, mlp, latent_dims, model_dict, WQ_all, W_all, WQ_adm, W_adm, grid_dims,
                 dropout=None, is_train=True, device='cpu'):
        super(MarioVAELModel, self).__init__(encoder=encoder, decoder=decoder, mlp=mlp, latent_dims=latent_dims,
                                             model_dict=model_dict, dropout=dropout, is_train=is_train,
                                             device=device)

        self.grid_dims = grid_dims
        # P(q,c) -> constraints in the query -> 81 worlds -> to compute the query prob
        self.WQ_all = WQ_all  # worlds x queries
        self.W_all = W_all  # worlds x facts
        # P(w|c) -> constraints as evidence -> 24 worlds -> sampling for decoder label to sample only admissible worlds given the constraint
        self.WQ_adm = WQ_adm  # worlds x queries
        self.W_adm = W_adm  # worlds x facts
        self.eps = 1e-5

        if W_all is not None:
            assert self.W_all.shape == torch.Size([81, 18]), self.W_all.shape
        if WQ_all is not None:
            assert self.WQ_all.shape == torch.Size([81, 5]), self.WQ_all.shape
        if W_adm is not None:
            assert self.W_adm.shape == torch.Size([24, 18]), self.W_adm.shape
        if WQ_adm is not None:
            assert self.WQ_adm.shape == torch.Size([24, 4]), self.WQ_adm.shape

    def forward(self, x, moves):

        # Input splitting
        x1, x2 = torch.split(x, x.shape[2] // 2, dim=2)

        # Image encoding
        mu1, logvar1 = self.encoder(x1)
        mu2, logvar2 = self.encoder(x2)

        # Sample z
        z1 = self.reparametrize(mu1, logvar1)
        z2 = self.reparametrize(mu2, logvar2)

        # Sub-symbolical component
        self.z_subsym1 = z1[:, self.latent_dim_sym:]
        self.z_subsym2 = z2[:, self.latent_dim_sym:]

        # Dropout on sub-symbolical variable
        if self.dropout:
            self.z_subsym1 = nn.Dropout(p=self.dropout)(self.z_subsym1)
            self.z_subsym2 = nn.Dropout(p=self.dropout)(self.z_subsym2)

        # Extract probability for each fact from the symbolical component of the latent space
        # The agent positions are mutually exclusive at each time step-> softmax
        self.facts_probs1 = nn.Softmax(dim=1)(self.mlp(z1[:, :self.latent_dim_sym]))
        self.facts_probs2 = nn.Softmax(dim=1)(self.mlp(z2[:, :self.latent_dim_sym]))
        self.facts_probs1 = torch.clip(self.facts_probs1, min=self.eps, max=0.9999)
        self.facts_probs2 = torch.clip(self.facts_probs2, min=self.eps, max=0.9999)
        facts = torch.cat([self.facts_probs1, self.facts_probs2], dim=1)

        # Compute worlds' probability P(W)=prod(p_i)prod(1-p_i)
        self.worlds_probs = self.compute_worlds_prob_conj(facts)

        # Computing query probability (query = "move" & constraints)
        self.query_prob = self.compute_query_prob(moves, self.worlds_probs)

        # Compute unnormalized logprob for the admissible worlds P(W|constraint)
        self.adm_worlds_logprob = self.admissible_worlds_logprob(facts)  # [batch_size, 24]

        # Sample world given P(W|constraint)
        self.world = F.gumbel_softmax(logits=self.adm_worlds_logprob, tau=1, hard=True)

        # Represent the sampled admissible world in the Herbrand base (truth value for the 18 facts)
        world_h = torch.matmul(self.world, self.W_adm.type(torch.float))

        # Split the world in the two
        world1, world2 = torch.split(world_h, self.mlp.n_facts // 2, dim=-1)
        # Image decoding
        x_recon1 = self.decode(self.z_subsym1, world1)
        x_recon2 = self.decode(self.z_subsym2, world2)

        return x1.permute(0,2,3,1), x_recon1.permute(0,2,3,1), mu1, logvar1, x2.permute(0,2,3,1), x_recon2.permute(0,2,3,1), mu2, logvar2, self.query_prob


    def define_herbrand_base(self, n_facts):
        """Defines the herbrand base to encode ProbLog worlds"""
        n_pos = n_facts // 2
        eye = torch.eye(n_pos)
        herbrand_base = torch.cat(
            [torch.cat((eye[i].expand(n_pos, n_pos), eye), dim=1) for i in
             range(n_pos)], dim=0)

        return herbrand_base

    def compute_worlds_prob_conj(self, probs):
        """
        Extracts the probability distributions of all the possible worlds.
        """
        # TODO: implement a more general method for not ADs facts, but also independent facts.
        log_p_i = torch.log(probs) @ self.W_all.T
        worlds_probs = torch.exp(log_p_i)
        return worlds_probs

    def admissible_worlds_logprob(self, probs):
        """
        Extracts the unnormalized log probability of those worlds that are admissible (i.e. satisfy the constraint).
        """
        log_p_i = torch.log(probs) @ self.W_adm.T
        log_q_i = torch.log(1 - probs) @ (1 - self.W_adm).T
        worlds_logprobs = log_p_i + log_q_i
        return worlds_logprobs

    def compute_query_prob(self, labels, worlds_probs):
        """
        Computes the query probability given worlds probability P(w) = prod(p_i)prod(1-p_i).
        The query is defined as the conjunction between the label and the constraint, to inject additional knowledge in the model.
        """
        p_w_model_q = torch.empty(labels.shape[0]).to(self.device)

        # Compute probability of all possible worlds
        p_all_worlds = torch.sum(worlds_probs, dim=1) + self.eps

        # Add the constraints query as true to the labels
        # so that the worlds must model both the specific label and the constraints.
        new_labels = torch.cat([labels, torch.ones(labels.shape[0], 1).to(self.device)], dim=1)

        # Compute probability of all possible worlds that model the query
        for i in range(len(labels)):
            w_q = torch.all(self.WQ_all == new_labels[i], axis=1)
            p_w_model_q[i] = torch.sum(w_q * worlds_probs[i], dim=0)

        # Query prob P(q)
        p_q = p_w_model_q / p_all_worlds

        return p_q

    def compute_query_prob_cond(self, q, worlds_probs):
        # TODO: implement a more general method for not ADs facts, but also independent facts for task generalization.
        # Probability of all possible worlds
        # sum(p(w|e))
        p_e = torch.sum(worlds_probs, dim=0) + self.eps

        # Compute probability of all possible worlds that model the query
        w_q = self.WQ_adm[:, int(q)]
        p_q_and_e = torch.sum(w_q * worlds_probs, dim=0)

        # Query prob P(q|e) = p
        p_q_given_e = p_q_and_e / p_e

        return p_q_given_e

    def compute_conditioned_worlds_prob(self, probs, evidence):
        # TODO
        evidence = evidence.to(self.device)
        # Compute the worlds probs given the evidence
        log_p_i = torch.log(probs) @ self.W_adm.T
        # Compute worlds probability
        p_w = torch.exp(log_p_i)
        # Create a mask over the worlds that satisfy the evidence
        w_q = torch.all((self.WQ_adm == torch.tensor(evidence)), axis=1)
        # Compute the probability of the worlds given the evidence
        p_w_and_e = (p_w * w_q) + self.eps
        p_e = torch.sum(p_w * w_q) + self.eps
        p_w_given_e = p_w_and_e / p_e

        return p_w_given_e

    def encode_image(self, x):

        with torch.no_grad():
            mu, logvar = self.encoder(x)
            # Sample z
            z = self.reparametrize(mu, logvar)

            # Sub-symbolical component
            z_subsym = z[:, self.latent_dim_sym:]

            # Extract probability for each fact from the symbolical component of the latent space
            facts_probs_ = nn.Softmax(dim=1)(self.mlp(z[:, :self.latent_dim_sym]))

            # Crisp probabilities
            facts_probs = torch.zeros_like(facts_probs_)
            mask = facts_probs_ >= facts_probs_.max()
            facts_probs[mask] = facts_probs_[mask]
            facts_probs[:, torch.where(facts_probs > 0.5)[1]] = 1.

        return z_subsym, facts_probs
