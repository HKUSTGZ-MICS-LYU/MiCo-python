import torch

from torch.utils.data import DataLoader
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