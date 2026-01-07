import os
import warnings
import zipfile
import urllib.request

import numpy as np
import torch

from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset

def mnist(batch_size=32, num_works = 0, shuffle = True, augment = True, resize: int = 28):

    transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST(root='data', train=True, transform=transform, download=True)
    test_data = datasets.MNIST(root='data', train=False, transform=transform)

    if augment:
        # Data Augmentation
        augmented_transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            # 10,10 seems to be best combination
            transforms.RandomRotation(degrees=10),  
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        augmented_train_data = datasets.MNIST(root='data', train=True, transform=augmented_transform)
        train_data = ConcatDataset([train_data, augmented_train_data])

    train_loader = DataLoader(train_data, batch_size, shuffle=shuffle, num_workers=num_works)
    test_loader = DataLoader(test_data, batch_size, shuffle=shuffle, num_workers=num_works)
    return train_loader, test_loader

def fashion_mnist(batch_size=32, num_works = 0, shuffle = True, augment = True):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_data = datasets.FashionMNIST(root='data', train=True, transform=transform, download=True)
    test_data = datasets.FashionMNIST(root='data', train=False, transform=transform)
    if augment:
        # Data Augmentation
        augmented_transform = transforms.Compose([
            transforms.RandomRotation(degrees=10),  
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor()
        ])
        augmented_train_data = datasets.FashionMNIST(root='data', train=True, transform=augmented_transform)
        train_data = ConcatDataset([train_data, augmented_train_data])

    train_loader = DataLoader(train_data, batch_size, shuffle=shuffle, num_workers=num_works)
    test_loader = DataLoader(test_data, batch_size, shuffle=shuffle, num_workers=num_works)
    return train_loader, test_loader

def cifar10(batch_size=32, num_works = 0, shuffle = True, augment = True):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR10(root='data', train=False, transform=test_transform)

    if augment:
        # Data Augmentation
        augmented_transform = transforms.Compose([
            transforms.RandomRotation(degrees=10),  
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            normalize
        ])
        augmented_train_data = datasets.CIFAR10(root='data', train=True, transform=augmented_transform)
        train_data = ConcatDataset([train_data, augmented_train_data])

    train_loader = DataLoader(train_data, batch_size, shuffle=shuffle, num_workers=num_works)
    test_loader = DataLoader(test_data, batch_size, shuffle=shuffle, num_workers=num_works)
    return train_loader, test_loader

def cifar100(batch_size=32, num_works = 0, shuffle = True, augment = True):

    mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_data = datasets.CIFAR100(root='data', train=True, download=True, transform=train_transform)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_data = datasets.CIFAR100(root='data', train=False, transform=test_transform)

    if augment:
        # Data Augmentation
        augmented_transform = transforms.Compose([
            transforms.RandomRotation(degrees=10),  
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        augmented_train_data = datasets.CIFAR100(root='data', train=True, transform=augmented_transform)
        train_data = ConcatDataset([train_data, augmented_train_data])

    train_loader = DataLoader(train_data, batch_size, shuffle=shuffle, num_workers=num_works)
    test_loader = DataLoader(test_data, batch_size, shuffle=shuffle, num_workers=num_works)

    return train_loader, test_loader

def imagenet(batch_size=32, num_works = 0, shuffle = True, resize: int = 256, root = "data"):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_data = datasets.ImageNet(root=root, split="train", transform=train_transform)

    test_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_data = datasets.ImageNet(root=root, split="val", transform=test_transform)

    train_loader = DataLoader(train_data, batch_size, shuffle=shuffle,
                              num_workers=num_works)
    test_loader = DataLoader(test_data, batch_size, shuffle=shuffle,
                             num_workers=num_works)
    
    return train_loader, test_loader

def tinystories(max_seq_len, vocab_size, device, batch_size=32, num_works=0):
    from TinyStories import TinyStoriesTask
    train_loader = TinyStoriesTask.iter_batches(
        batch_size=batch_size, 
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        device=device,
        vocab_source="data/tinystories",
        split="train")
    
    test_loader = TinyStoriesTask.iter_batches(
        batch_size=batch_size, 
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        device=device,
        vocab_source="data/tinystories",
        split="val")
    
    return train_loader, test_loader


UCI_HAR_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
# Small epsilon to avoid division by zero when standard deviation is zero
UCI_HAR_EPS = 1e-6


def _ensure_uci_har(root: str) -> str:
    os.makedirs(root, exist_ok=True)
    extracted_root = os.path.join(root, "UCI HAR Dataset")
    if os.path.exists(extracted_root):
        return extracted_root
    archive_path = os.path.join(root, "uci_har.zip")
    if not os.path.exists(archive_path):
        urllib.request.urlretrieve(UCI_HAR_URL, archive_path)
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(root)
    return extracted_root


def uci_har(batch_size=64, num_workers=0, shuffle=True, root="data/uci_har", **kwargs):
    """
    Load the UCI Human Activity Recognition dataset with standardization.

    Args:
        batch_size: Batch size for data loaders.
        num_workers: Number of workers for PyTorch DataLoader.
        shuffle: Whether to shuffle training data.
        root: Root directory for dataset storage.
        num_works: Deprecated alias for num_workers accepted via kwargs. This will be removed in version 0.2.
    """
    num_works = kwargs.pop("num_works", None)
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")
    if num_works is not None:
        warnings.warn(
            "`num_works` is deprecated and will be removed in version 0.2; use `num_workers` instead.",
            FutureWarning,
        )
        num_workers = num_works
    dataset_root = _ensure_uci_har(root)
    x_train = np.loadtxt(os.path.join(dataset_root, "train", "X_train.txt"))
    y_train = np.loadtxt(os.path.join(dataset_root, "train", "y_train.txt")).astype(int) - 1

    x_test = np.loadtxt(os.path.join(dataset_root, "test", "X_test.txt"))
    y_test = np.loadtxt(os.path.join(dataset_root, "test", "y_test.txt")).astype(int) - 1

    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)

    mean = x_train.mean(0, keepdim=True)
    std = x_train.std(0, keepdim=True) + UCI_HAR_EPS
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    train_dataset = TensorDataset(x_train, torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(x_test, torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
