import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dataset import get_test_loader
from model import MiniResNet


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(seed):
    device = get_device()
    print("Using device:", device)
    model = MiniResNet().to(device)
    model.load_state_dict(
        torch.load(f"best_model_seed{seed}.pth", map_location=device)
    )
    test_dataset, test_loader = get_test_loader(batch_size=64)
    num_classes = len(test_dataset.classes)
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            counts = torch.bincount(
                labels * num_classes + predictions, minlength=num_classes**2
            )
            confusion_matrix += counts.reshape(num_classes, num_classes)

    accuracy = 100 * confusion_matrix.diag().sum().item() / confusion_matrix.sum().item()
    normalized = confusion_matrix.float() / confusion_matrix.sum(dim=1, keepdim=True)
    print(f"Test accuracy: {accuracy:.2f}%")

    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix)
    plt.xticks(range(num_classes), test_dataset.classes, rotation=45)
    plt.yticks(range(num_classes), test_dataset.classes)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Confusion Matrix")
    plt.colorbar()
    for row in range(num_classes):
        for column in range(num_classes):
            plt.text(
                column, row, round((normalized[row, column] * 100).item()),
                ha="center", va="center",
            )
    plt.tight_layout()
    plt.savefig(
        f"confusion_matrix_seed{seed}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a CIFAR-10 checkpoint.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    evaluate(args.seed)


if __name__ == "__main__":
    main()
