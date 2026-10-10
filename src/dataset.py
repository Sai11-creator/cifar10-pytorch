import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split,Subset
import matplotlib.pyplot as plt

def get_train_transform():
    return transforms.Compose([
        transforms.RandomCrop(
            32,
            padding = 4
        ),
        transforms.RandomHorizontalFlip(),        
        transforms.ToTensor(),
        transforms.Normalize (
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        ),
        transforms.RandomErasing(p=0.25)
    ])
def get_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])
    
def get_train_val_loader(batch_size=64):
    transform = get_train_transform()
    train_full_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=False,
        transform=transform
    )
    val_full_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=False,
        transform=get_test_transform()
    )
    indices = torch.randperm(
    50000,
    generator=torch.Generator().manual_seed(42))

    train_indices = indices[:45000]
    val_indices = indices[45000:]
    
    train_dataset=Subset(train_full_dataset,train_indices)
    val_dataset = Subset(val_full_dataset,val_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader=DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader,train_dataset,val_loader


def get_test_loader(batch_size=64):
    transform = get_test_transform()

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=False,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return test_dataset, test_loader





