import os.path
from itertools import product

import numpy as np
import torch
from problog.ddnnf_formula import DDNNF
from problog.formula import LogicFormula
from problog.program import PrologString
from tqdm import tqdm

import problog_model


def run_program(model):
    p = PrologString(model)
    lf = LogicFormula.create_from(p)  # ground the program
    ddnnf = DDNNF.create_from(lf)  # compile CNF to ddnnf
    return ddnnf.evaluate()


# Problog Model for base task
world_dims = problog_model.WORLD_DIMS
base_model = problog_model.BASE_MODEL
facts = problog_model.FACTS_BASE_MODEL
possible_queries = problog_model.BASE_QUERIES_WITH_CONSTRAINT

# Path to store the matrices
CWD = os.getcwd()
folder_path = os.path.join(CWD, 'problog_matrices', f'{world_dims[0]}x{world_dims[0]}')
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

#########################################
######  Computing matrix of worlds ######
#########################################
"""
We have two ADs made of 9 possible facts, thus we have 81 possible worlds.
"""
number_ads = 2
facts_per_ad = 9
W = []
# Build possible AD groundings
ad = np.eye(facts_per_ad, facts_per_ad, dtype=int)

for w in tqdm(list(product(*[ad.tolist() for _ in range(number_ads)]))):
    world = w[0] + w[1]
    W.append(world)

torch.save(W, os.path.join(folder_path, "W_all.pt"))

###############################################################
######  Computing worlds where query is satisfied worlds ######
###############################################################
"""
Here we simply go over the worlds and we check whether the query is satisfied or not;
we store a boolean map into the variable query vector.
"""

queries = [f"query({q})." for q in possible_queries]
model_query = base_model + "\n" + "\n".join(queries)
w_q = torch.zeros((len(W), len(possible_queries)))
# Iterate over the possible worlds
for i, possible_world in enumerate(tqdm(W), 0):
    current_world = facts.format(*possible_world)
    current_model = model_query.format(current_world)
    # Extract the probability for each query
    res = run_program(current_model)
    res = {str(key): value for key, value in res.items()}
    # If the query is satified, then store 1 for the corresponding world, otherwise 0
    for j, q in enumerate(possible_queries):
        w_q[i, j] = 1 if res[q] > 0.5 else 0

torch.save(w_q, os.path.join(folder_path, "WQ_all.pt"))

####################################################
######  Computing matrix of admissible worlds ######
####################################################

possible_queries = problog_model.BASE_QUERIES

"""
We have two ADs made of 9 possible facts, thus we have 81 possible worlds.
However, in the base task the agent_dict is not allowed to stay in the same position, neither to make more than one step.
Thus, we have to take into consideration only those worlds where the robot makes exactly one step, that are 24.
"""
number_ads = 2
facts_per_ad = 9
W = []
# Add constraint
model = base_model + "\nquery(constraint)."
# Build possible AD groundings
ad = np.eye(facts_per_ad, facts_per_ad, dtype=int)
#
for w in tqdm(list(product(*[ad.tolist() for _ in range(number_ads)]))):
    world = w[0] + w[1]
    current_world = facts.format(*world)
    current_model = model.format(current_world)
    m = list(run_program(current_model).items())[0][1]
    if m > 0.5:
        W.append(world)

torch.save(W, os.path.join(folder_path, "W_adm.pt"))

###############################################################
######  Computing worlds where query is satisfied worlds ######
###############################################################
"""
Here we simply go over the admissible worlds and we check whether the query is satisfied or not
and we store a boolean map into the variable query vector.
"""

queries = [f"query({q})." for q in possible_queries]
model_query = base_model + "\n" + "\n".join(queries)
w_q = torch.zeros((len(W), len(possible_queries)))
# Iterate over the possible worlds
for i, possible_world in enumerate(tqdm(W), 0):
    current_world = facts.format(*possible_world)
    current_model = model_query.format(current_world)
    # Extract the probability for each query
    res = run_program(current_model)
    res = {str(key): value for key, value in res.items()}
    # If the query is satified, then store 1 for the corresponding world, otherwise 0
    for j, q in enumerate(possible_queries):
        w_q[i, j] = 1 if res[q] > 0.5 else 0

torch.save(w_q, os.path.join(folder_path, "WQ_adm.pt"))
