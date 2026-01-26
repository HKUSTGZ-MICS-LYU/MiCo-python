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


def _speechcommands_label_to_index(walker):
    labels = sorted({os.path.basename(os.path.dirname(p)) for p in walker})
    return {label: idx for idx, label in enumerate(labels)}


def speechcommands(batch_size=64, num_works=0, 
                   shuffle=True, root="data/speechcommands",
                   preprocess: str = 'raw'):
    from torchaudio import transforms as T
    from torchaudio.datasets import SPEECHCOMMANDS

    # Check root path exists
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)

    class SubsetSC(SPEECHCOMMANDS):
        def __init__(self, subset: str = None):
            super().__init__(root, download=True)
            def load_list(filename):
                filepath = os.path.join(self._path, filename)
                with open(filepath) as f:
                    return [os.path.join(self._path, line.strip()) for line in f]
            if subset == "validation":
                self._walker = load_list("validation_list.txt")
            elif subset == "testing":
                self._walker = load_list("testing_list.txt")
            elif subset == "training":
                excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
                excludes = set(excludes)
                self._walker = [w for w in self._walker if w not in excludes]

    train_set = SubsetSC("training")
    val_set = SubsetSC("validation")
    test_set = SubsetSC("testing")

    label_to_index = _speechcommands_label_to_index(train_set._walker)

    target_sample_rate = 16000
    resampler = T.Resample(orig_freq=target_sample_rate, new_freq=target_sample_rate)

    def _preprocess(batch):
        waveforms = []
        labels = []
        for waveform, sample_rate, label, *_ in batch:
            if sample_rate != target_sample_rate:
                waveform = resampler(waveform)
            waveform = waveform.mean(dim=0, keepdim=True)
            length = waveform.shape[-1]
            if length < target_sample_rate:
                pad = target_sample_rate - length
                waveform = torch.nn.functional.pad(waveform, (0, pad))
            else:
                waveform = waveform[..., :target_sample_rate]
            waveforms.append(waveform)
            labels.append(label_to_index[label])
        if preprocess == 'spectrogram':
            spectrogram_transform = T.MelSpectrogram(
                sample_rate=target_sample_rate,
                n_mels=40
            )
            waveforms = [spectrogram_transform(waveform) for waveform in waveforms]
        elif preprocess == 'mfcc':
            mfcc_transform = T.MFCC(
                sample_rate=target_sample_rate,
                n_mfcc=64,
                melkwargs={"n_mels": 64}
            )
            waveforms = [mfcc_transform(waveform) for waveform in waveforms]
            # Reshape: (batch, n_mfcc, time)
            # waveforms = [waveform.squeeze(0) for waveform in waveforms]
        return torch.stack(waveforms), torch.tensor(labels, dtype=torch.long)

    train_loader = DataLoader(
        train_set, batch_size, shuffle=shuffle, 
        num_workers=num_works, collate_fn=_preprocess)
    test_loader = DataLoader(
        test_set, batch_size, shuffle=False, 
        num_workers=num_works, collate_fn=_preprocess)

    return train_loader, test_loader


# if __name__ == "__main__":
#     train_loader, test_loader = speechcommands(
#         batch_size=32, num_works=1, shuffle=True, preprocess='mfcc'
#     )
#     for batch in train_loader:
#         waveforms, labels = batch
#         print(waveforms.shape, labels.shape)
#         break


# =============================================================================
# HuggingFace Datasets for LLM Evaluation
# =============================================================================

def wikitext(
    batch_size: int = 8,
    max_seq_len: int = 512,
    tokenizer=None,
    variant: str = "wikitext-2-raw-v1",
    num_workers: int = 0,
    shuffle: bool = False,
    root: str = "data",
    **kwargs,
):
    """
    Load WikiText dataset for language model perplexity evaluation.
    
    WikiText provides a standard benchmark for language model evaluation.
    This function returns DataLoaders that yield (input_ids, labels) tuples
    suitable for next-token prediction tasks.
    
    Args:
        batch_size: Batch size for data loaders
        max_seq_len: Maximum sequence length for tokenization
        tokenizer: HuggingFace tokenizer (if None, uses GPT-2 tokenizer)
        variant: WikiText variant ("wikitext-2-raw-v1", "wikitext-103-raw-v1")
        num_workers: Number of workers for DataLoader
        shuffle: Whether to shuffle data
        root: Root directory for caching (not used, HF handles caching)
        **kwargs: Additional arguments (for compatibility)
        
    Returns:
        train_loader, test_loader: DataLoader objects
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets is required for WikiText. "
            "Install with: pip install datasets"
        )
    
    # Handle deprecated num_works argument
    num_works = kwargs.pop("num_works", None)
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")
    if num_works is not None:
        warnings.warn(
            "`num_works` is deprecated and will be removed in version 0.2; use `num_workers` instead.",
            FutureWarning,
        )
        num_workers = num_works
    
    # Load tokenizer if not provided
    if tokenizer is None:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "HuggingFace transformers is required for WikiText. "
                "Install with: pip install transformers"
            )
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Load WikiText dataset
    dataset = load_dataset("wikitext", variant)
    
    def tokenize_and_chunk(examples):
        """Tokenize and chunk text into fixed-length sequences."""
        # Concatenate all texts
        texts = [text for text in examples["text"] if text.strip()]
        if not texts:
            return {"input_ids": [], "labels": []}
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        
        # Concatenate all tokens
        all_tokens = []
        for tokens in tokenized["input_ids"]:
            all_tokens.extend(tokens)
        
        # Chunk into fixed-length sequences
        chunks = []
        for i in range(0, len(all_tokens) - max_seq_len, max_seq_len):
            chunk = all_tokens[i:i + max_seq_len + 1]  # +1 for labels
            if len(chunk) == max_seq_len + 1:
                chunks.append(chunk)
        
        if not chunks:
            return {"input_ids": [], "labels": []}
        
        # Create input_ids and labels
        input_ids = [chunk[:-1] for chunk in chunks]
        labels = [chunk[1:] for chunk in chunks]
        
        return {"input_ids": input_ids, "labels": labels}
    
    # Process datasets
    train_dataset = dataset["train"].map(
        tokenize_and_chunk,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing train set",
    )
    
    test_dataset = dataset["test"].map(
        tokenize_and_chunk,
        batched=True,
        remove_columns=dataset["test"].column_names,
        desc="Tokenizing test set",
    )
    
    # Set format for PyTorch
    train_dataset.set_format(type="torch", columns=["input_ids", "labels"])
    test_dataset.set_format(type="torch", columns=["input_ids", "labels"])
    
    def collate_fn(batch):
        """Collate function for DataLoader."""
        input_ids = torch.stack([item["input_ids"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return input_ids, labels
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    return train_loader, test_loader


def wikitext2(batch_size: int = 8, max_seq_len: int = 512, tokenizer=None, **kwargs):
    """
    Load WikiText-2 dataset.
    
    WikiText-2 is a smaller version (~2M tokens) suitable for quick evaluation.
    """
    return wikitext(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        tokenizer=tokenizer,
        variant="wikitext-2-raw-v1",
        **kwargs,
    )


def wikitext103(batch_size: int = 8, max_seq_len: int = 512, tokenizer=None, **kwargs):
    """
    Load WikiText-103 dataset.
    
    WikiText-103 is a larger version (~103M tokens) for more comprehensive evaluation.
    """
    return wikitext(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        tokenizer=tokenizer,
        variant="wikitext-103-raw-v1",
        **kwargs,
    )


def hf_text_dataset(
    dataset_name: str,
    batch_size: int = 8,
    max_seq_len: int = 512,
    tokenizer=None,
    text_column: str = "text",
    split_train: str = "train",
    split_test: str = "test",
    num_workers: int = 0,
    shuffle: bool = False,
    subset: str = None,
    **kwargs,
):
    """
    Generic loader for HuggingFace text datasets.
    
    This function provides a flexible way to load any text dataset from
    HuggingFace Hub for language model evaluation.
    
    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "wikitext", "c4", "openwebtext")
        batch_size: Batch size for data loaders
        max_seq_len: Maximum sequence length for tokenization
        tokenizer: HuggingFace tokenizer (if None, uses GPT-2 tokenizer)
        text_column: Name of the text column in the dataset
        split_train: Name of the training split
        split_test: Name of the test split
        num_workers: Number of workers for DataLoader
        shuffle: Whether to shuffle data
        subset: Dataset subset/configuration (e.g., "wikitext-2-raw-v1")
        **kwargs: Additional arguments passed to load_dataset
        
    Returns:
        train_loader, test_loader: DataLoader objects
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets is required. "
            "Install with: pip install datasets"
        )
    
    # Load tokenizer if not provided
    if tokenizer is None:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "HuggingFace transformers is required. "
                "Install with: pip install transformers"
            )
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    if subset:
        dataset = load_dataset(dataset_name, subset, **kwargs)
    else:
        dataset = load_dataset(dataset_name, **kwargs)
    
    def tokenize_and_chunk(examples):
        """Tokenize and chunk text into fixed-length sequences."""
        texts = [text for text in examples[text_column] if text and text.strip()]
        if not texts:
            return {"input_ids": [], "labels": []}
        
        tokenized = tokenizer(
            texts,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        
        all_tokens = []
        for tokens in tokenized["input_ids"]:
            all_tokens.extend(tokens)
        
        chunks = []
        for i in range(0, len(all_tokens) - max_seq_len, max_seq_len):
            chunk = all_tokens[i:i + max_seq_len + 1]
            if len(chunk) == max_seq_len + 1:
                chunks.append(chunk)
        
        if not chunks:
            return {"input_ids": [], "labels": []}
        
        input_ids = [chunk[:-1] for chunk in chunks]
        labels = [chunk[1:] for chunk in chunks]
        
        return {"input_ids": input_ids, "labels": labels}
    
    # Process datasets
    train_dataset = dataset[split_train].map(
        tokenize_and_chunk,
        batched=True,
        remove_columns=dataset[split_train].column_names,
        desc=f"Tokenizing {split_train} set",
    )
    
    test_dataset = dataset[split_test].map(
        tokenize_and_chunk,
        batched=True,
        remove_columns=dataset[split_test].column_names,
        desc=f"Tokenizing {split_test} set",
    )
    
    train_dataset.set_format(type="torch", columns=["input_ids", "labels"])
    test_dataset.set_format(type="torch", columns=["input_ids", "labels"])
    
    def collate_fn(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return input_ids, labels
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    return train_loader, test_loader