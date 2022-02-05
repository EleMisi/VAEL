from itertools import product
from random import sample

import numpy as np
import torch
from torch import tensor, load
from torch.utils.data import Dataset

from utils.nMNIST_image_generator import mnist_sequence


# TODO: Documentation

class nMNIST(Dataset):
    """nMNIST dataset."""

    def __init__(self, sequence_len, worlds=None, digits=10, batch_size=32, idxs=None, train=False, sup=False,
                 sup_digits=None, sup_digits_per_world=2, label2idx=None, data_path="./data/MNIST/2mnist_3digits.npz"):
        """
        Args:
            sequence_len (int):
            batch_size (int, optional):
            train (bool, optional):
            sup (bool, optional):
        """
        self.n_digits = digits
        self.sup_digits_x_world = sup_digits_per_world
        self.images, self.labels = self.read_data(path=data_path, train=train)
        self.mean, self.std = self.images.mean(), self.images.std()

        if worlds:
            self.worlds = worlds
        else:
            self.worlds = list(product(range(digits), repeat=sequence_len))
        self.batch_size = batch_size

        if label2idx:
            self.label2idx = label2idx
        else:
            # Create dictionary of indexes for each world
            self.label2idx = {c: set() for c in self.worlds}
            for k, v in self.label2idx.items():
                print(k)
                for i, label in enumerate(self.labels):
                    if tuple(label[:2]) == k:
                        v.add(i)
        if idxs:
            self.load_idxs(idxs)
            self.len = int(np.ceil(len(self.worlds) * self.samples_x_world / self.batch_size))
            self.world_counter = {c: self.samples_x_world // self.batch_size for c in self.worlds}
            # self.world_counter = {c: self.samples_x_world for c in self.worlds}
            # Supervised mode, define 10 fixed supervised images over the 28800 of the training set
            if sup and not sup_digits:
                self.sup_digits = {c: sample(self.idxs[c].ravel().tolist(), sup_digits_per_world) for c in self.worlds}
                print("NEW supervised digits idx:", self.sup_digits)
            elif sup and sup_digits:
                self.sup_digits = sup_digits
                print("LOAD supervised digits idx:", self.sup_digits)
            else:
                self.sup_digits = None

        else:
            self.create_idxs(train, sup)

    def load_idxs(self, idxs):
        """
        Load the list of image indexes for reproducibility
        """
        self.idxs = idxs
        self.samples_x_world = self.idxs[self.worlds[0]].shape[0] * self.idxs[self.worlds[0]].shape[1]
        self.n_samples = len(self.idxs) * self.samples_x_world
        print("Indexes loaded: total images {}, samples per world {}".format(self.n_samples, self.samples_x_world))

    def create_idxs(self, train=True, sup=False):
        """
        Create list of image indexes for reproducibility.
        """

        print("No indexes specified...")
        if train:
            n_samples = int(input(
                "Please specify the desired number of samples in the dataset (80% training set, 20% validation set):\n"))
            self.n_samples = {}
            self.n_samples['train'] = int(.8 * n_samples)
            self.n_samples['val'] = int(.2 * n_samples)
            self.idxs = {'train': {c: set() for c in self.worlds},
                         'val': {c: set() for c in self.worlds}}
            for dataset in ['train', 'val']:
                x = self.n_samples[dataset] // len(self.worlds)
                m = int(self.batch_size)
                n = ((x - 1) | (m - 1)) + 1
                self.samples_x_world = int(n)
                self.n_samples[dataset] = self.samples_x_world * len(self.worlds)
                assert n % m == 0, "Invalid samples per worlds!"
                print(dataset, "set samples per world:", self.samples_x_world)
                # Create list of indexes for reproducibility
                self.idxs[dataset] = {c: set() for c in self.worlds}
                for c in self.idxs[dataset]:
                    print(c)
                    i = 0
                    while len(self.idxs[dataset][c]) < self.samples_x_world:
                        print(len(self.idxs[dataset][c]), self.samples_x_world)
                        i += 1
                        self.idxs[dataset][c].add(sample(self.label2idx[c], 1)[0])

                # Reshape
                for k in self.idxs[dataset]:
                    self.idxs[dataset][k] = tensor(list(self.idxs[dataset][k])).reshape(-1, self.batch_size)
                # Reset world counter
                self.reset_counter()
                print("Indexes created.")

            if sup:
                self.sup_digits = {c: sample(self.idxs['train'][c].ravel().tolist(), self.sup_digits_x_world) for c in
                                   self.worlds}
                print("NEW supervised digits idx:", self.sup_digits)

        else:
            self.n_samples = int(input("Please specify the desired number of samples in the test set:\n"))
            x = self.n_samples // len(self.worlds)
            self.samples_x_world = int(x)
            self.n_samples = self.samples_x_world * len(self.worlds)
            print("Test set samples per world:", self.samples_x_world)
            # Create list of indexes for reproducibility
            self.idxs = {c: set() for c in self.worlds}
            for c in self.idxs:
                print(c)
                i = 0
                while len(self.idxs[c]) < self.samples_x_world:
                    print(len(self.idxs[c]), self.samples_x_world)
                    i += 1
                    self.idxs[c].add(sample(self.label2idx[c], 1)[0])
            self.batch_size = self.samples_x_world
            # Reshape
            for k in self.idxs:
                self.idxs[k] = tensor(list(self.idxs[k])).reshape(-1, self.batch_size)

            # Reset world counter
            self.reset_counter()
            print("Indexes created.")

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
            return images.reshape(self.batch_size, 28, 56), np.array(labels).reshape(self.batch_size, 4)

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
            train_imgs, train_labels = mnist_sequence(n_digit=self.n_digits, sequence_len=2, samples_x_class=1000,
                                                      train=True, download=True)
            test_imgs, test_labels = mnist_sequence(n_digit=self.n_digits, sequence_len=2, samples_x_class=100,
                                                    train=False, download=True)
            data = {'train': {'images': train_imgs, 'labels': train_labels},
                    'test': {'images': test_imgs, 'labels': test_labels}}
            torch.save(data, './data/MNIST/2mnist_{0}digits/2mnist_{0}digits.pt'.format(self.n_digits))
            print("Data saved in ./data/MNIST/2mnist_{0}digits/2mnist_{0}digits.pt'".format(self.n_digits))

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
