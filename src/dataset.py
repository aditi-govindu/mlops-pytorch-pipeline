# Import modules.
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_transforms(train: bool = True) -> transforms.Compose:
    '''Apply data transformations to images to be compatible.
    Args:
        train: Boolean flag indicating transformation must be applied
        if the flag is true. 
        Use images as they are if not used for training.

    Returns:
        transformed image array. 
    '''
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])

def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    '''Function to load CIFAR10 dataset for model training.
    Args:
        data_dir: Location to read data from.
        batch_size: No. of items to read at any point.
        num_workers: Total no. of parallel threads that can run on the dataset.
    
    Returns:
        tuple: train and validation data loader.
    '''
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=get_transforms(train=True)
    )
    val_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=get_transforms(train=False)
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader