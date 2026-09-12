import os
import torch
import torch.nn as nn
import torch.optim as optim

from agents.disease_agent.src.dataset import create_dataloaders
from agents.disease_agent.src.model import BaselineCNN


# Config
EPOCHS = 5
LR = 0.001
NUM_CLASSES = 5
MODEL_PATH = "agents/disease_agent/models/baseline_cnn_best.pth"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Data
train_loader, val_loader, test_loader = create_dataloaders()

# Model
model = BaselineCNN(num_classes=NUM_CLASSES).to(device)

# Loss + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Create model directory
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

best_val_acc = 0

# Training
for epoch in range(EPOCHS):

    # Train
    model.train()
    train_correct = 0
    train_total = 0
    train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += labels.size(0)

    train_acc = 100 * train_correct / train_total

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss / len(train_loader):.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss / len(val_loader):.4f} | "
        f"Val Acc: {val_acc:.2f}%"
    )

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc

        torch.save(model.state_dict(), MODEL_PATH)

        print("  ✓ Best model saved")


# Test
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

test_acc = 100 * correct / total

print("\n" + "=" * 50)
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"Test Accuracy:            {test_acc:.2f}%")
print(f"Model saved:              {MODEL_PATH}")
print("=" * 50)