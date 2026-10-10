import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn, optim

from dataset import get_train_val_loader
from model import MiniResNet


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


def train(seed, num_epochs=50):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = get_device()
    model = MiniResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    train_loader, _, val_loader = get_train_val_loader(batch_size=64)

    train_accuracies, val_accuracies = [], []
    best_accuracy = 0.0
    checkpoint = f"best_model_seed{seed}.pth"

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        average_loss = running_loss / len(train_loader)
        train_accuracy = accuracy(model, train_loader, device)
        val_accuracy = accuracy(model, val_loader, device)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), checkpoint)
            torch.save(model.state_dict(), "best_model.pth")

        print(
            f"Epoch {epoch + 1}/{num_epochs} | Loss: {average_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, train_accuracies, label="Train")
    plt.plot(epochs, val_accuracies, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.savefig("accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train MiniResNet on CIFAR-10.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.seed)


if __name__ == "__main__":
    main()
