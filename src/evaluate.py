import torch
import matplotlib.pyplot as plt

from dataset import get_test_loader
from model import MiniResNet

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)


model = MiniResNet().to(device)


model.load_state_dict(torch.load("best_model.pth"))
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
            (torch.round((normalized_confusion_matrix[i, j]) * 100, decimals=0)).item(),
            ha="center",
            va="center"
        )
plt.show()
