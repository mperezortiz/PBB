import torch
import numpy as np
import torch.optim as optim
import torch.distributions as td
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler


def loaddataset(name, noise=0.2):
    torch.manual_seed(7)
    # several choices of datasets (e.g., with or without noise)

    if name == 'mnist-multiclass':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train = datasets.MNIST(
            'mnist-data/', train=True, download=True, transform=transform)
        test = datasets.MNIST(
            'mnist-data/', train=False, download=True, transform=transform)

    elif name == 'mnist-binary':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        target_transform = transforms.Lambda(lambda x: 0 if x <= 4 else 1)
        train = datasets.MNIST(
            'mnist-data/', train=True, download=True, transform=transform, target_transform=target_transform)
        test = datasets.MNIST(
            'mnist-data/', train=False, download=True, transform=transform, target_transform=target_transform)

    elif name == 'mnist-noise':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        target_transform = transforms.Lambda(
            lambda x: np.random.randint(0, 9) if np.random.rand() <= noise else x)
        train = datasets.MNIST(
            'mnist-data/', train=True, download=True, transform=transform, target_transform=target_transform)
        test = datasets.MNIST(
            'mnist-data/', train=False, download=True, transform=transform)

    elif name == 'cifar10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2023, 0.1994, 0.2010)),
             ])
        train = datasets.CIFAR10(
            './data', train=True, download=True, transform=transform)
        test = datasets.CIFAR10(
            './data', train=False, download=True, transform=transform)

    else:
        assert False
    return train, test


def loadbatches(train, test, loader, batch_size, prior=False, perc_train=1.0, perc_prior=0.2):

    ntrain = len(train.data)
    ntest = len(test.data)

    if prior == False:
        indices = list(range(ntrain))
        split = int(np.round((perc_train)*ntrain))
        random_seed = 10
        np.random.seed(random_seed)
        np.random.shuffle(indices)

        train_idx = indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)

        set_bound_1batch = torch.utils.data.DataLoader(
            train, batch_size=len(train_idx), sampler=train_sampler, **loader)
        test_1batch = torch.utils.data.DataLoader(
            test, batch_size=ntest, shuffle=True, **loader)
        train_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, sampler=train_sampler, **loader)
        test_loader = torch.utils.data.DataLoader(
            test, batch_size=batch_size, shuffle=True, **loader)
        prior_loader = None
        set_val_bound = train_loader

    else:
        # reduce training data if needed
        new_num_train = int(np.round((perc_train)*ntrain))
        indices = list(range(new_num_train))
        split = int(np.round((perc_prior)*new_num_train))
        random_seed = 10
        np.random.seed(random_seed)
        np.random.shuffle(indices)

        all_train_sampler = SubsetRandomSampler(indices)
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        set_bound_1batch = torch.utils.data.DataLoader(
            train, batch_size=len(train_idx), sampler=train_sampler, **loader)
        set_val_bound = torch.utils.data.DataLoader(
            train, batch_size=batch_size, sampler=train_sampler, shuffle=False)
        test_1batch = torch.utils.data.DataLoader(
            test, batch_size=ntest, shuffle=True, **loader)
        train_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, sampler=all_train_sampler, shuffle=False)
        prior_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, sampler=valid_sampler, shuffle=False)
        test_loader = torch.utils.data.DataLoader(
            test, batch_size=batch_size, shuffle=True, **loader)

    # train_loader comprises all the data used in training and prior_loader the data used to build
    # the prior
    # set_bound_1batch and set_bound are the set of data points used to evaluate the bound.
    # the only difference between these two is that onf of them is splitted in multiple batches
    # while the 1batch one is only one batch. This is for computational efficiency with some
    # of the large architectures used.
    # The same is done for test_1batch
    return train_loader, test_loader, prior_loader, set_bound_1batch, test_1batch, set_val_bound
