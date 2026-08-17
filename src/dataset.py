import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_dataloaders(dataset_name: str = 'CIFAR10', 
                    data_dir: str = './data', batch_size: int = 64, val_split: float = 0.1):
    '''Function to load data if CIFAR10 or FMNIST dataset is used.

    Args:
        dataset_name: CIFAR10 or FMNIST datasets are supported currently.
        data_dir: Location to read data from. ./data is default.
        batch_size: No. of images read in 1 set. 64 is default.
        val_split: Amount of data used for validation. 10% is default.

    Returns:
        Train, test and validation datasets.
    '''

    if dataset_name == 'CIFAR10':
        # Data transformations for training and validation.
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        full_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size

        # Get train, test and validation datasets for CIFAR10.
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        val_dataset.dataset.transform = transform_val
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_val)
    elif dataset_name == "FashionMNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        full_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Load the final datasets: 80-10-10 for train-validation-testing.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader