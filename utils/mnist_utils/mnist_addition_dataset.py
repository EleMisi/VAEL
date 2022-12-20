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
    """nMNIST dataset class"""

    def __init__(self, sequence_len, worlds=None, digits=10, batch_size=32, idxs=None, mode='train', sup=False,
                 sup_digits=None, sup_samples_per_world=2, data_path=None):
        """
        @param sequence_len: number of digits in the sequence
        @param worlds: digits pairs to consider
        @param digits: digits values (e.g., digits = 3 means digits 0,1,2)
        @param batch_size: number of images in a batch
        @param idxs: list of image indexes for reproducibility
        @param mode: string that specify the data to be loaded: training ('train'), validation ('val') or testing ('test') data
        @param sup: whether to give direct supervision on digits
        @param sup_digits: list of supervised digits for reproducibility
        @param sup_samples_per_world: number of supervised samples per pair of digits
        @param data_path: path to dataset
        """
        self.n_digits = digits
        self.mode = mode
        self.sup_samples_per_world = sup_samples_per_world
        self.images, self.labels = self.read_data(path=data_path)
        self.mean, self.std = self.images.mean(), self.images.std()
        if worlds:
            self.worlds = worlds
        else:
            self.worlds = list(product(range(digits), repeat=sequence_len))
        self.batch_size = batch_size
        self.load_idxs(idxs)
        self.len = int(np.ceil(len(self.worlds) * self.samples_x_world / self.batch_size))
        self.world_counter = {c: self.samples_x_world // self.batch_size for c in self.worlds}
        # Supervised mode
        if sup and not sup_digits:
            self.sup_digits = {c: sample(self.idxs[c].ravel().tolist(), self.sup_samples_per_world) for c in
                               self.worlds}
            print("NEW supervised digits idx:", self.sup_digits)
        elif sup and sup_digits:
            self.sup_digits = sup_digits
            print("LOAD supervised digits idx:", self.sup_digits)
        else:
            self.sup_digits = None

    def load_idxs(self, idxs):
        """
        Load the list of image indexes for reproducibility and computes the samples per pair accordingly
        @param idxs: list of image indexes for reproducibility
        """
        self.samples_x_world = idxs[self.worlds[0]].shape[0]
        self.idxs = {k: v.reshape(-1, self.batch_size) for k, v in idxs.items()}
        self.n_samples = len(self.idxs) * self.samples_x_world
        print("Indexes loaded: total images {}, samples per world {}".format(self.n_samples, self.samples_x_world))

    def get_image(self, idx):
        """
        Given the index, returns the corresponding image and its label. The label is in the form [digit1, ..., digitN, sum of digits]
        @param idx: index of the desired image
        @return: image and label as array
        """

        # Get image at index idx an image
        image = self.images[idx].astype(float)
        image = (image - self.mean) / self.std

        label = self.labels[idx]
        label = np.concatenate((label, -1), axis=None)

        return np.array(image), np.array(label)

    def __len__(self):
        """
        @return: length of dataset in terms of number of batches
        """
        return int(np.ceil(len(self.worlds) * self.samples_x_world / self.batch_size))

    def __getitem__(self, _):
        """
        @return: batch of images and labels
        """

        images = []
        labels = []
        # Remaining pairs of digits
        remaining_worlds = [c for c in self.worlds if self.world_counter[c] > 0]
        if remaining_worlds == []:
            raise "TODO: implement error for exhausted generator"
        else:
            # Randomly sampling a pair c of digits from the remaining ones
            c = sample(remaining_worlds, 1)[0]
            self.world_counter[c] -= 1
            # Load the batch of images and labels corresponding to pair c
            for i, idx in enumerate(self.idxs[c][self.world_counter[c]]):
                img, label = self.get_image(idx)
                images.append(img)
                labels.append(label)

            # If direct supervision on digits is required,
            # replace one image in the current batch with a fully supervised image
            if self.sup_digits:
                i = sample(range(len(images)), 1)[0]
                sup_id = sample(self.sup_digits[c], 1)
                sup_img, sup_label = self.get_image(sup_id)
                sup_label[-1] = i
                images[i] = sup_img.reshape(28, 56)
                labels[i] = sup_label

            images = np.array(images)
            return images.reshape(-1, 28, 56), np.array(labels).reshape(-1, 4)

    def read_data(self, path):
        """
        Returns images and labels stored in 'path'.
        @param path: path where data are stored
        @return: images and labels, if they exist.
        """
        try:
            print("Loading data...")
            data = load(path)
            print("Loaded.")
        except:
            print("No dataset found.")

        images = data[self.mode]['images']
        labels = data[self.mode]['labels']

        return images, labels

    def reset_counter(self):
        """
        Resets dataset counter.
        """
        self.world_counter = {c: self.samples_x_world // self.batch_size for c in self.worlds}
        # self.world_counter = {c: self.samples_x_world for c in self.worlds}


def create_sample(X, target_sequence, digit2idx):
    """
    Creates the required nMNIST sample.
    @param X: MNIST data (images and labels)
    @param target_sequence: sequence of digits to create
    @param digit2idx: dictionary of MNIST digits and corresponding indexes in X
    @return: nMNIST image and label as arrays, and the corresponding index
    """
    idxs = [choice(digit2idx[digit][0]) for digit in target_sequence]
    imgs = [X[idx] for idx in idxs]
    new_image = np.concatenate(imgs, axis=1)
    new_label = target_sequence + (np.sum(target_sequence),)

    return new_image, np.array(new_label).astype('int32'), idxs


def create_dataset(n_digit=2, sequence_len=2, samples_x_world=100, train=True, download=False):
    """
    Creates the required nMNIST dataset.
    @param n_digit: number of digits values to be consider (e.g., with digits = 3 we'll have only digits 0,1,2)
    @param sequence_len: number of digits in a sequence
    @param samples_x_world: number of samples per each sequence
    @param train: whether to load MNIST train or test dataset
    @param download: whether to download MNIST dataset
    @return: nMNIST images and labels as arrays, and corresponding indexes as dictionary
    """
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
    """Checks whether the dataset exists, if not it creates the required data.
    @param n_digits: number of digits values to be consider (e.g., with digits = 3 we'll have only digits 0,1,2)
    @param data_folder: dataset folder
    @param data_file: dataset file name
    @param dataset_dim: dictionary with desired number of samples for training ('train'), validation ('val') and testing
     ('test') set. The dimensions will be redefined to ensure the same samples for each pair of digits
    """
    Path(data_folder).mkdir(parents=True, exist_ok=True)
    data_path = os.path.join(data_folder, data_file)
    try:
        load(data_path)
    except:
        print("No dataset found -> building a new dataset")
        # Define dataset dimension so to have the same number of worlds
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
