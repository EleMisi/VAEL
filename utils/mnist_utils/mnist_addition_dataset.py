import os
from itertools import product
from pathlib import Path
from random import sample, choice

import numpy as np
import torch
from torch import tensor, load
from torch.utils.data import Dataset
from torchvision import datasets
from tqdm import tqdm


# TODO: Documentation

class nMNIST(Dataset):
    """nMNIST dataset."""

    def __init__(self, sequence_len, worlds=None, digits=10, batch_size=32, idxs=None, train=False, sup=False,
                 sup_digits=None, sup_digits_per_world=2, data_path=None):
        self.n_digits = digits
        self.sup_digits_x_world = sup_digits_per_world
        self.images, self.labels = self.read_data(path=data_path, train=train)
        self.mean, self.std = self.images.mean(), self.images.std()
        if worlds:
            self.worlds = worlds
        else:
            self.worlds = list(product(range(digits), repeat=sequence_len))
        self.batch_size = batch_size
        self.load_idxs(idxs)
        self.len = int(np.ceil(len(self.worlds) * self.samples_x_world / self.batch_size))
        self.world_counter = {c: self.samples_x_world // self.batch_size for c in self.worlds}
        # Supervised mode: select n = sup_digits_per_world images for a full supervision
        if sup and not sup_digits:
            self.sup_digits = {c: sample(self.idxs[c].ravel().tolist(), sup_digits_per_world) for c in self.worlds}
            print("NEW supervised digits idx:", self.sup_digits)
        elif sup and sup_digits:
            self.sup_digits = sup_digits
            print("LOAD supervised digits idx:", self.sup_digits)
        else:
            self.sup_digits = None

    def load_idxs(self, idxs):
        """
        Load the list of image indexes for reproducibility
        """
        self.samples_x_world = idxs[self.worlds[0]].shape[0]
        self.idxs = {k: v.reshape(self.batch_size, -1) for k, v in idxs.items()}
        self.n_samples = len(self.idxs) * self.samples_x_world
        print("Indexes loaded: total images {}, samples per world {}".format(self.n_samples, self.samples_x_world))

    def get_image(self, idx):
        """
        Create the target sequence by randomly sampling the required digits from X.
        Return the new image and its label in the form [digit1, ..., digitN, sum of digits]
        """

        # Get image at index idx an image
        image = self.images[idx].astype(float)
        image = (image - self.mean) / self.std
        # image = image / 255.
        label = self.labels[idx]
        label = np.concatenate((label, -1), axis=None)

        return np.array(image), np.array(label)

    def __len__(self):
        return int(np.ceil(len(self.worlds) * self.samples_x_world / self.batch_size))

    def __getitem__(self, _):

        images = []
        labels = []
        remaining_worlds = [c for c in self.worlds if self.world_counter[c] > 0]
        if remaining_worlds == []:
            raise "TODO: implement error for exhausted generator"
        else:
            c = sample(remaining_worlds, 1)[0]  # Randomly sampling 1 sequence of digits
            self.world_counter[c] -= 1
            for i, idx in enumerate(self.idxs[c][self.world_counter[c]]):
                img, label = self.get_image(idx)  # Build the corresponding image and label
                images.append(img)
                labels.append(label)

            # Replace an image with supervised digits in each batch
            if self.sup_digits:
                i = sample(range(len(images)), 1)[0]
                sup_id = sample(self.sup_digits[c], 1)
                sup_img, sup_label = self.get_image(sup_id)
                sup_label[-1] = i
                images[i] = sup_img.reshape(28, 56)
                labels[i] = sup_label

            images = np.array(images)
            return images.reshape(-1, 28, 56), np.array(labels).reshape(-1, 4)

    def read_data(self, path, train=True):
        """
        Returns images and labels
        """
        try:
            print("Loading data...")
            data = load(path)
            print("Loaded.")
        except:
            print("No dataset found.")

        if train:
            images = data['train']['images']
            labels = data['train']['labels']
        else:
            images = data['test']['images']
            labels = data['test']['labels']

        return images, labels

    def reset_counter(self):
        self.world_counter = {c: self.samples_x_world // self.batch_size for c in self.worlds}
        # self.world_counter = {c: self.samples_x_world for c in self.worlds}


def create_sample(X, target_sequence, digit2idx):
    idxs = [choice(digit2idx[digit][0]) for digit in target_sequence]
    imgs = [X[idx] for idx in idxs]
    new_image = np.concatenate(imgs, axis=1)
    new_label = target_sequence + (np.sum(target_sequence),)

    return new_image, np.array(new_label).astype('int32'), idxs


def create_dataset(n_digit=2, sequence_len=2, samples_x_world=100, train=True, download=False):
    # Download data
    MNIST = datasets.MNIST(root='./data/raw/', train=train, download=download)

    x, y = MNIST.data, MNIST.targets

    # Create dictionary of indexes for each digit
    digit2idx = {k: [] for k in range(10)}
    for k, v in digit2idx.items():
        v.append(np.where(y == k)[0])

    # Create the list of all possible permutations with repetition of 'sequence_len' digits
    worlds = list(product(range(n_digit), repeat=sequence_len))
    imgs = []
    labels = []

    # Create data sample for each class
    for c in tqdm(worlds):
        for i in range(samples_x_world):
            img, label, idxs = create_sample(x, c, digit2idx)
            imgs.append(img)
            labels.append(label)

    # Create dictionary of indexes for each world
    label2idx = {c: set() for c in worlds}
    for k, v in tqdm(label2idx.items()):
        for i, label in enumerate(labels):
            if tuple(label[:2]) == k:
                v.add(i)
    label2idx = {k: tensor(list(v)) for k, v in label2idx.items()}

    return np.array(imgs).astype('int32'), np.array(labels), label2idx


def check_dataset(n_digits, data_folder, data_file, dataset_dim):
    """Checks whether the dataset exists, if not creates it."""
    Path(data_folder).mkdir(parents=True, exist_ok=True)
    data_path = os.path.join(data_folder, data_file)
    try:
        load(data_path)
    except:
        print("No dataset found.")
        # Define dataset dimension so to have teh same number of worlds
        n_worlds = n_digits * n_digits
        samples_x_world = {k: int(d / n_worlds) for k, d in dataset_dim.items()}
        dataset_dim = {k: s * n_worlds for k, s in samples_x_world.items()}

        train_imgs, train_labels, train_indexes = create_dataset(n_digit=n_digits, sequence_len=2,
                                                                 samples_x_world=samples_x_world['train'], train=True,
                                                                 download=True)
        val_imgs, val_labels, val_indexes = create_dataset(n_digit=n_digits, sequence_len=2,
                                                           samples_x_world=samples_x_world['val'], train=True,
                                                           download=True)
        test_imgs, test_labels, test_indexes = create_dataset(n_digit=n_digits, sequence_len=2,
                                                              samples_x_world=samples_x_world['test'], train=False,
                                                              download=True)

        print(
            f"Dataset dimensions: \n\t{dataset_dim['train']} train ({samples_x_world['train']} samples per world), \n\t{dataset_dim['val']} validation ({samples_x_world['val']} samples per world), \n\t{dataset_dim['test']} test ({samples_x_world['test']} samples per world)")

        data = {'train': {'images': train_imgs, 'labels': train_labels},
                'val': {'images': val_imgs, 'labels': val_labels},
                'test': {'images': test_imgs, 'labels': test_labels}}

        indexes = {'train': train_indexes,
                   'val': val_indexes,
                   'test': test_indexes}

        torch.save(data, data_path)
        for key, value in indexes.items():
            torch.save(value, os.path.join(data_folder, f'{key}_indexes.pt'))

        print(f"Dataset saved in {data_folder}")
