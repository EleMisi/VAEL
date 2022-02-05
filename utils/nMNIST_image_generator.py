from itertools import product
from random import choice

import numpy as np
from torchvision import datasets


def create_sample(X, target_sequence, digit2idx):
    idxs = [choice(digit2idx[digit][0]) for digit in target_sequence]
    imgs = [X[idx] for idx in idxs]
    new_image = np.concatenate(imgs, axis=1)
    new_label = target_sequence + (np.sum(target_sequence),)

    return new_image, np.array(new_label).astype('int32'), idxs


def mnist_sequence(n_digit=2, sequence_len=2, samples_x_class=100, train=True, download=False, indexes=None):
    # Download data
    MNIST = datasets.MNIST(root='./data/raw/', train=train, download=download)

    x, y = MNIST.data, MNIST.targets

    # Create dictionary of indexes for each digit
    digit2idx = {k: [] for k in range(10)}
    for k, v in digit2idx.items():
        v.append(np.where(y == k)[0])

    if indexes is None:
        print("NO indexes.")
        # Create the list of all possible permutations with repetition of 'sequence_len' digits
        classes = product(range(n_digit), repeat=sequence_len)
        imgs = []
        indexes = []
        labels = []

        # Create data sample for each class
        for c in classes:
            for i in range(samples_x_class):
                print('Generating sample {}/{} of class {}'.format(i + 1, samples_x_class, c))
                img, label, idxs = create_sample(x, c, digit2idx)
                imgs.append(img)
                labels.append(label)
                indexes.append(idxs)
    else:
        print("Loaded indexes.")
        imgs = []
        labels = []
        for idxs in indexes:
            img = [x[idx] for idx in idxs]
            label = [y[idx] for idx in idxs]
            label = label + [np.sum(label)]
            img = np.concatenate(img, axis=1)
            imgs.append(img)
            labels.append(label)

    return np.array(imgs).astype('int32'), np.array(labels), np.array(indexes)
