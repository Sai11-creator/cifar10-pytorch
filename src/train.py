import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import get_train_val_loader, get_test_loader
from model import MiniResNet

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

model = MiniResNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.StepLR(
    optimizer=optimizer,
    step_size = 5,
    gamma=0.5
)

def evaluate(model, loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy

train_loader,train_dataset,val_loader = get_train_val_loader(batch_size=64)



model.train()

#Train
train_losses = []
train_accuracies = []
val_accuracies = []
num_epochs = 15

best_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    scheduler.step()
    average_loss = running_loss / len(train_loader)

        
    train_accuracy = evaluate(model, train_loader)
    val_accuracy = evaluate(model,val_loader)
    
    
    train_losses.append(average_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    if val_accuracy>best_accuracy:
        best_accuracy=val_accuracy
        torch.save(
            model.state_dict(),
            "best_model.pth"
        )
    
    print(
        f"Epoch {epoch + 1}/{num_epochs} | "
        f"Loss: {average_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Acc: {val_accuracy:.2f}%"
    )
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch + 1} | "
        f"Loss: {average_loss:.4f} | "
        f"LR: {current_lr:.6f}"
        )
 
 
 
torch.save(model.state_dict(), "model.pth")

epochs = range(1, num_epochs + 1)

plt.plot(epochs, train_accuracies, label="Train")
plt.plot(epochs, val_accuracies, label="Validation")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()