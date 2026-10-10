import torch
import matplotlib.pyplot as plt

from dataset import get_test_loader
from model import MiniResNet

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)


model = MiniResNet().to(device)


model.load_state_dict(torch.load(f"best_model_seed{args.seed}.pth",map_location=device))
test_dataset, test_loader = get_test_loader(batch_size=64)


num_classes = 10
confusion_matrix = torch.zeros(
    num_classes,
    num_classes,
    dtype=torch.int64
)
model.eval()
with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, dim=1)

        predicted = predicted.cpu()

        for true_label, predicted_label in zip(labels, predicted):

            confusion_matrix[
                true_label.item(),
                predicted_label.item()
            ] += 1
correct = confusion_matrix.diag().sum().item()
total = confusion_matrix.sum().item()
test_accuracy = 100 * correct / total
normalized_confusion_matrix = (
    confusion_matrix.float()
    / confusion_matrix.sum(dim=1, keepdim=True)
)


print(f"Test accuracy: {test_accuracy:.2f}%")
plt.figure(figsize=(10, 8))
plt.imshow(confusion_matrix)
plt.xticks(
    range(10),
    test_dataset.classes,
    rotation=45
)
plt.yticks(
    range(10),
    test_dataset.classes
)
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.title("Confusion Matrix")
plt.colorbar()
plt.tight_layout()
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(
            j,
            i,
            round((normalized_confusion_matrix[i, j] * 100).item()),
            ha="center",
            va="center"
        )
plt.savefig(f"confusion_matrix_seed{args.seed}.png",
    dpi=150,
    bbox_inches="tight")
plt.close()
