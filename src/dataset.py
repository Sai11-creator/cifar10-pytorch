import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


DATA_ROOT = "./data"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def get_train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25),
    ])


def get_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_train_val_loader(batch_size=64):
    train_data = datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=False, transform=get_train_transform()
    )
    val_data = datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=False, transform=get_test_transform()
    )
    indices = torch.randperm(50_000, generator=torch.Generator().manual_seed(42))
    train_dataset = Subset(train_data, indices[:45_000])
    val_dataset = Subset(val_data, indices[45_000:])

    loader_options = {"batch_size": batch_size, "num_workers": 4, "pin_memory": True}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    return train_loader, train_dataset, val_loader


def get_test_loader(batch_size=64):
    test_dataset = datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=False, transform=get_test_transform()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return test_dataset, test_loader
