
import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from agents.disease_agent.src.dataset import (
    create_dataloaders,
    CLASS_NAMES,
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 5
EPOCHS = 5
LEARNING_RATE = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_DIR = "agents/disease_agent/models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "resnet18_best.pth"
)


# ============================================================
# DATA
# ============================================================

print("Loading dataset...")

train_loader, val_loader, test_loader = create_dataloaders()

print(f"Device: {DEVICE}")
print(f"Classes: {CLASS_NAMES}")

print(f"Training batches:   {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches:       {len(test_loader)}")


# ============================================================
# LOAD PRETRAINED RESNET18
# ============================================================

print("\nLoading pretrained ResNet18...")

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)


# ============================================================
# REPLACE CLASSIFIER
# ============================================================

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    NUM_CLASSES
)

model = model.to(DEVICE)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

best_val_acc = 0.0

best_model_weights = copy.deepcopy(
    model.state_dict()
)


for epoch in range(EPOCHS):
    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Statistics
        running_loss += (
            loss.item() * images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    train_loss = running_loss / total
    train_acc = 100.0 * correct / total


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss_total = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss_total += (
                loss.item() * images.size(0)
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_loss = val_loss_total / val_total

    val_acc = (
        100.0 * val_correct / val_total
    )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.2f}%"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_acc > best_val_acc:
        best_val_acc = val_acc

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            {
                "model_state_dict": best_model_weights,
                "class_names": CLASS_NAMES,
                "num_classes": NUM_CLASSES,
                "best_val_accuracy": best_val_acc,
            },
            MODEL_PATH
        )

        print(
            f"  ✓ Best ResNet18 model saved "
            f"({best_val_acc:.2f}%)"
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    best_model_weights
)

print("\nTraining complete.")

print(
    f"Best Validation Accuracy: "
    f"{best_val_acc:.2f}%"
)

print(
    f"Model saved to: {MODEL_PATH}"
)


# ============================================================
# TEST EVALUATION
# ============================================================

model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        test_total += labels.size(0)

        test_correct += (
            predicted == labels
        ).sum().item()


test_accuracy = (
    100.0 * test_correct / test_total
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n========================================")
print("        RESNET18 TEST RESULTS")
print("========================================")

print(
    f"Best Validation Accuracy: "
    f"{best_val_acc:.2f}%"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy:.2f}%"
)

print("========================================")
