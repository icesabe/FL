#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gzip
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

"""
-------
MNIST
-------
"""

DATASET_FOLDER = "dataset/MNIST/"


class MnistDataset(Dataset):
    """Convert the MNIST pkl file into a Pytorch Dataset"""

    def __init__(self, file_path, k):
        dataset = pickle.load(open(file_path, "rb"))

        self.X = dataset[0][k]
        self.Y = np.array(dataset[1][k])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 3D input 1x28x28
        x = torch.Tensor([self.X[idx]]) / 255  #image
        y = torch.LongTensor([self.Y[idx]])[0]  #label

        return x, y

class RawMnistDataset(Dataset):
    def __init__(self, folder, data_name, label_name,transform=None):
        """
        images_path='train-images-idx3-ubyte'
        labels_path='train-labels-idx1-ubyte'
        """
        (train_data, train_labels) = load_data(folder, data_name, label_name)
        self.train_data = train_data
        self.train_labels = train_labels
        self.transform = transform

    def __getitem__(self, index):

        img, target = self.train_data[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_data)


def load_data(data_folder, data_name, label_name):
    """RawMnistDataset function"""

    with gzip.open(os.path.join(data_folder,label_name), 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8) #MNIST

    with gzip.open(os.path.join(data_folder,data_name), 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    return (x_train, y_train)


class MnistShardDataset(Dataset):
    """Convert the MNIST pkl file into a Pytorch Dataset"""

    def __init__(self, file_path, k):

        with open(file_path, "rb") as pickle_file:
            dataset = pickle.load(pickle_file)
            self.features = np.vstack(dataset[0][k])

            vector_labels = list()
            for idx, digit in enumerate(dataset[1][k]):
                vector_labels += [digit] * len(dataset[0][k][idx])
                #check cluster sampling code: py_func.read_db.py

            self.labels = np.array(vector_labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # 3D input 1x28x28
        x = torch.Tensor([self.features[idx]]) / 255
        y = torch.LongTensor([self.labels[idx]])[0]

        return x, y


def get_1shard(ds, row_0: int, digit: int, samples: int):
    """return an array from `ds` of `digit` starting of `row_0` in the indices of `ds`"""

    row = row_0

    shard = list()

    while len(shard) < samples:
        if ds.train_labels[row] == digit:
            shard.append(ds.train_data[row].numpy())
        row += 1

    return row, shard
    # row index after extraction, list of extracted samples


def clients_set_MNIST_shard(file_name, n_clients, batch_size=100, shuffle=True):
    """Download for all the clients their respective dataset"""

    print(file_name)

    list_dl = list()
    for k in range(n_clients):
        dataset_object = MnistShardDataset(file_name, k)
        dataset_dl = DataLoader(
            dataset_object, batch_size=batch_size, shuffle=shuffle
        )
        list_dl.append(dataset_dl)

    return list_dl


def create_MNIST_ds_1shard_per_client(n_clients, samples_train, samples_test):
    '''Createing 10 shards each class with n_clients'''
    MNIST_train = datasets.MNIST(root=DATASET_FOLDER, train=True, download=True)
    MNIST_test = datasets.MNIST(root=DATASET_FOLDER, train=False, download=True)

    shards_train, shards_test = [], []
    labels = []

    for i in range(10):
        row_train, row_test = 0, 0
        for j in range(10):
            row_train, shard_train = get_1shard(
                MNIST_train, row_train, i, samples_train
            )
            row_test, shard_test = get_1shard(
                MNIST_test, row_test, i, samples_test
            )

            shards_train.append([shard_train])
            shards_test.append([shard_test])

            labels += [[i]]

    X_train = np.array(shards_train)
    X_test = np.array(shards_test)

    y_train = labels
    y_test = y_train

    folder = DATASET_FOLDER
    train_path = f"MNIST_shard_train_{n_clients}_{samples_train}.pkl"
    with open(folder + train_path, "wb") as output:
        pickle.dump((X_train, y_train), output)

    test_path = f"MNIST_shard_test_{n_clients}_{samples_test}.pkl"
    with open(folder + test_path, "wb") as output:
        pickle.dump((X_test, y_test), output)


def partition_MNIST_dataset(
        dataset,
        file_name: str,
        matrix,
        n_clients: int,
        n_classes: int,
        train: bool,
):
    """Partition dataset into `n_clients`.
    Each client i has matrix[k, i] of data of class k"""

    list_clients_X = [[] for i in range(n_clients)]
    list_clients_y = [[] for i in range(n_clients)]

    client_sample_nums = [600] * n_clients

    list_idx = []

    for k in range(n_classes):
        idx_k = np.where(np.array(dataset.train_labels) == k)[0]
        list_idx += [idx_k]


    for idx_client, n_sample in enumerate(client_sample_nums):

        clients_idx_i = []
        client_samples = 0

        for k in range(n_classes):

            if k < 9:
                samples_digit = int(matrix[idx_client, k] * n_sample)
            if k == 9:
                samples_digit = n_sample - client_samples
            client_samples += samples_digit

            clients_idx_i = np.concatenate(
                (clients_idx_i, np.random.choice(list_idx[k], samples_digit))
            )

        clients_idx_i = clients_idx_i.astype(int)

        for idx_sample in clients_idx_i:
            list_clients_X[idx_client] += [dataset.train_data[idx_sample]]
            list_clients_y[idx_client] += [dataset.train_labels[idx_sample]]

        list_clients_X[idx_client] = np.array(list_clients_X[idx_client])

    folder = DATASET_FOLDER
    print("len:", np.shape(list_clients_X), np.shape(list_clients_y))
    # len: (100, 600) (100, 600)  n_clients, n_samples
    with open(folder + file_name, "wb") as output:
        pickle.dump((list_clients_X, list_clients_y), output)


def create_MNIST_dirichlet(
        dataset_name: str,
        patition: str,
        alpha: float,
        n_clients: int,
        n_classes: int,
):
    """Create a CIFAR dataset partitioned according to a
    dirichilet distribution Dir(alpha)"""

    from numpy.random import dirichlet

    # shape ``(size, k)``
    matrix = dirichlet([alpha] * n_classes, size=n_clients) #n_clients, n_classes
    # if matrix.isnull().any():
    #     matrix.replace(np.nan, 0)  # remove nan
    print("matrix:", np.shape(matrix))
    ####Values represent the proportion of each class in each client's data####
    # MNIST_train = datasets.MNIST(
    #     root=DATASET_FOLDER,
    #     train=True,
    #     download=True,
    #     transform=transforms.ToTensor(),
    # )
    MNIST_train = RawMnistDataset(
        DATASET_FOLDER+'MNIST/raw', "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", transform=transforms.ToTensor())

    # MNIST_test = datasets.MNIST(
    #     root=DATASET_FOLDER,
    #     train=False,
    #     download=True,
    #     transform=transforms.ToTensor(),
    # )
    MNIST_test = RawMnistDataset(
        DATASET_FOLDER+'MNIST/raw', "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", transform=transforms.ToTensor())

    file_name_train = f"{dataset_name}_{patition}_train_{n_clients}.pkl"
    partition_MNIST_dataset(
        MNIST_train,
        file_name_train,
        matrix,
        n_clients,
        n_classes,
        True,
    )

    file_name_test = f"{dataset_name}_{patition}_test_{n_clients}.pkl"
    partition_MNIST_dataset(
        MNIST_test,
        file_name_test,
        matrix,
        n_clients,
        n_classes,
        False,
    )


def clients_set_MNIST(
        file_name: str, n_clients: int, batch_size: int, shuffle=True
):
    """Download for all the clients their respective dataset"""

    print(file_name)

    list_dl = list()

    for k in range(n_clients):
        dataset_object = MnistDataset(file_name, k)
        # Convert the MNIST pkl file into image, label
        dataset_dl = DataLoader(
            dataset_object, batch_size=batch_size, shuffle=shuffle
        )

        list_dl.append(dataset_dl)

    return list_dl


def get_MNIST_dataloaders(dataset_name, patition, batch_size: int, shuffle=True, reset_data=False):
    """
    """

    if patition == "iid":

        n_clients = 100
        samples_train, samples_test = 600, 100

        mnist_trainset = datasets.MNIST(
            root=DATASET_FOLDER,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        mnist_train_split = torch.utils.data.random_split(
            mnist_trainset, [samples_train] * n_clients
        )
        list_dls_train = [
            torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
            for ds in mnist_train_split
        ]

        mnist_testset = datasets.MNIST(
            root=DATASET_FOLDER,
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        mnist_test_split = torch.utils.data.random_split(
            mnist_testset, [samples_test] * n_clients
        )
        list_dls_test = [
            torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
            for ds in mnist_test_split
        ]

    elif patition == "shard":
        n_clients = 100
        samples_train, samples_test = 500, 80

        file_name_train = f"MNIST_shard_train_{n_clients}_{samples_train}.pkl"
        path_train = DATASET_FOLDER + file_name_train

        file_name_test = f"MNIST_shard_test_{n_clients}_{samples_test}.pkl"
        path_test = DATASET_FOLDER + file_name_test

        if not os.path.isfile(path_train):
            create_MNIST_ds_1shard_per_client(
                n_clients, samples_train, samples_test
            )

        list_dls_train = clients_set_MNIST_shard(
            path_train, n_clients, batch_size=batch_size, shuffle=shuffle
        )

        list_dls_test = clients_set_MNIST_shard(
            path_test, n_clients, batch_size=batch_size, shuffle=shuffle
        )

    # dirichlet
    elif patition[0:4] == "dir_":
        n_classes = 10
        n_clients = 100
        alpha = patition[4:] #alpha = 0.1 means "dir_0.1"

        file_name_train = f"{dataset_name}_{patition}_train_{n_clients}.pkl"
        path_train = DATASET_FOLDER + file_name_train

        file_name_test = f"{dataset_name}_{patition}_test_{n_clients}.pkl"
        path_test = DATASET_FOLDER + file_name_test

        # fix the dataset
        if not os.path.isfile(path_train):
            print("creating dataset alpha:", alpha)
            create_MNIST_dirichlet(
                dataset_name, patition, alpha, n_clients, n_classes
            )

        list_dls_train = clients_set_MNIST(
            path_train, n_clients, batch_size, True
        )

        list_dls_test = clients_set_MNIST(
            path_test, n_clients, batch_size, True
        )

    return list_dls_train, list_dls_test

#train_loaders, test_loaders = get_MNIST_dataloaders(
#    dataset_name="MNIST",
#    partition="dir_0.5",
#    batch_size=64,
#    shuffle=True
#)