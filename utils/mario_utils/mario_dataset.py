import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.mario_utils import problog_model
from utils.mario_utils.create_mario_dataset import  create_mario_dataset

class MarioDataset(Dataset):
    """Karel Dataset"""

    def __init__(self, path, mode, batch_size=32, seed=123456):
        self.mode = mode
        self.rnd_gen = np.random.default_rng(seed)
        self.batch_size = batch_size
        # Check if dataset exists
        self.check_dataset(path)
        self.moves, self.labels, self.images, self.pos, self.agents, self.targets, self.bkgs, self.frames, self.idxs, self.mean, self.std = self.load_data(
            path=path)
        self.vec2move = {tuple(v.tolist()): k for k, v in self.move2vec.items()}
        self.remaining_idxs = self.idxs
        self.n_samples = len(self.moves)
        self.end = False

    def __len__(self):
        return len(self.images)

    def __getitem__(self, _):
        try:
            idxs = self.rnd_gen.choice(self.remaining_idxs, size=self.batch_size, replace=False, shuffle=False)
        except ValueError as ve:
            idxs = self.rnd_gen.choice(self.remaining_idxs, size=len(self.remaining_idxs), replace=False, shuffle=False)
        label = self.labels[idxs]
        pos1 = self.pos[idxs][:, 0]
        pos2 = self.pos[idxs][:, 1]
        image1 = (self.images[idxs][:, 0] - self.mean) / self.std
        image2 = (self.images[idxs][:, 1] - self.mean) / self.std
        agent = self.agents[idxs]
        target = self.targets[idxs]
        bkg = self.bkgs[idxs]
        frame = self.frames[idxs]
        # Encoding
        pos1 = [self.pos2label[tuple(a.tolist())] for a in pos1]
        pos2 = [self.pos2label[tuple(a.tolist())] for a in pos2]
        agent = [self.agents2label[a] for a in agent]
        target = [self.targets2label[a] for a in target]
        bkg = [self.bkgs2label[a] for a in bkg]
        frame = [self.frames2label[a] for a in frame]
        # Update remaining idxs
        self.remaining_idxs = [idx for idx in self.remaining_idxs if idx not in idxs]
        if len(self.remaining_idxs) == 0:
            self.end = True
        # print(idxs)
        # print(len(self.remaining_idxs))
        batch = {'pos1': pos1,
                 'pos2': pos2,
                 'labels': label,
                 'imgs1': image1,
                 'imgs2': image2,
                 'agents': agent,
                 'targets': target,
                 'bkgs': bkg,
                 'frames': frame}
        return batch

    def load_data(self, path):
        """
        Load mode from the specified path.
        Returns the loaded list of moves and list of paired images.
        """
        path = os.path.join(path, '3x3')
        # Load images
        images = np.load(os.path.join(path, f'{self.mode}_images.npy'))
        # Compute dataset mean and std
        mean, std = np.mean(images), np.std(images)
        # Load moves
        moves = np.load(os.path.join(path, f'{self.mode}_moves.npy'))
        # Load positions
        pos = np.load(os.path.join(path, f'{self.mode}_pos.npy'))
        # Load styles info
        frames = np.load(os.path.join(path, f'{self.mode}_frames.npy'))
        agents = np.load(os.path.join(path, f'{self.mode}_agents.npy'))
        bkgs = np.load(os.path.join(path, f'{self.mode}_bkgs.npy'))
        targets = np.load(os.path.join(path, f'{self.mode}_targets.npy'))
        # Define query to vector encoding
        queries = problog_model.BASE_QUERIES
        query2vec = {move: np.zeros(len(set(moves))) for move in set(moves)}
        for move in query2vec:
            for i in range(len(queries)):
                if move in queries[i]:
                    query2vec[move][i] = 1.
        self.move2vec = query2vec
        # Build labels from string moves
        labels = np.zeros(shape=(images.shape[0], len(queries)), dtype='uint8')
        for j in range(moves.shape[0]):
            for i in range(len(queries)):
                if moves[j] in queries[i]:
                    labels[j][i] = 1
        # Shuffle idxs
        idxs = list(range(images.shape[0]))
        idxs = self.rnd_gen.choice(idxs, size=images.shape[0], replace=False, shuffle=True)
        # Check consistency
        assert len(moves) == len(images)
        # Convert to Tensor
        labels = torch.tensor(labels, dtype=torch.uint8)
        images = torch.tensor(images, dtype=torch.uint8)
        pos = torch.tensor(pos, dtype=torch.uint8)
        # Define unique features
        unique_pos = set()
        for position in pos[:, 0, :].tolist():
            unique_pos.add(tuple(position))
        unique_pos = list(unique_pos)
        unique_targets = list(set(targets))
        unique_agents = list(set(agents))
        unique_bkgs = list(set(bkgs))
        unique_frames = list(set(frames))
        # Sort
        unique_targets.sort()
        unique_agents.sort()
        unique_bkgs.sort()
        unique_frames.sort()
        unique_pos.sort()
        # Build lkup-tables
        self.targets2label = {unique_targets[i]: i for i in range(len(unique_targets))}
        self.agents2label = {unique_agents[i]: i for i in range(len(unique_agents))}
        self.bkgs2label = {unique_bkgs[i]: i for i in range(len(unique_bkgs))}
        self.frames2label = {unique_frames[i]: i for i in range(len(unique_frames))}
        self.pos2label = {(0, 0): 0,
                         (1, 0): 1,
                         (2, 0): 2,
                         (2, 1): 3,
                         (1, 1): 4,
                         (0, 1): 5,
                         (0, 2): 6,
                         (1, 2): 7,
                         (2, 2): 8}

        return moves, labels, images, pos, agents, targets, bkgs, frames, idxs, mean, std,

    def reset(self, shuffle):
        """Reset the remaining idxs list and shuffle it, if required"""
        if shuffle:
            self.idxs = self.rnd_gen.choice(self.idxs, size=len(self.idxs), replace=False, shuffle=True)
        self.remaining_idxs = self.idxs
        self.end = False



    def check_dataset(self, path):
        """Checks whether the dataset exists, if not creates it."""

        try:
            self.load_data(path=path)
        except:
            print(f"No dataset found in {path}. Creating Mario dataset...")
            create_mario_dataset(path)

            print(f"Dataset saved in {path}")