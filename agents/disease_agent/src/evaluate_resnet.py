import os
import torch
import torch.nn as nn
from torchvision import models

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)

from agents.disease_agent.src.dataset import (
    create_dataloaders,
    CLASS_NAMES,
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = os.path.join(
    "agents",
    "disease_agent",
    "models",
    "resnet18_best.pth"
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

train_loader, val_loader, test_loader = create_dataloaders()

print(f"Device: {DEVICE}")
print(f"Classes: {CLASS_NAMES}")
print(f"Test batches: {len(test_loader)}")


# ============================================================
# CREATE RESNET18 ARCHITECTURE
# ============================================================

print("\nCreating ResNet18 model...")

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)


# ============================================================
# REPLACE FINAL CLASSIFIER
# ============================================================

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    NUM_CLASSES
)

model = model.to(DEVICE)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print(f"\nLoading trained model from:")
print(MODEL_PATH)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


# ============================================================
# LOAD MODEL WEIGHTS
# ============================================================

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print("✓ ResNet18 model loaded successfully")


# ============================================================
# DISPLAY SAVED INFORMATION
# ============================================================

if "best_val_accuracy" in checkpoint:

    print(
        f"Best Validation Accuracy: "
        f"{checkpoint['best_val_accuracy']:.2f}%"
    )

if "class_names" in checkpoint:

    print(
        f"Saved Classes: "
        f"{checkpoint['class_names']}"
    )


# ============================================================
# EVALUATION
# ============================================================

model.eval()

all_labels = []
all_predictions = []

print("\nRunning evaluation on test dataset...")

with torch.no_grad():

    for images, labels in test_loader:

        # Move images and labels to GPU/CPU
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # ----------------------------------------------------
        # FORWARD PASS
        # ----------------------------------------------------

        outputs = model(images)

        # ----------------------------------------------------
        # GET PREDICTED CLASS
        # ----------------------------------------------------

        _, predicted = torch.max(
            outputs,
            1
        )

        # ----------------------------------------------------
        # STORE ACTUAL AND PREDICTED LABELS
        # ----------------------------------------------------

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predicted.cpu().numpy()
        )


# ============================================================
# ACCURACY
# ============================================================

test_accuracy = accuracy_score(
    all_labels,
    all_predictions
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    target_names=CLASS_NAMES,
    digits=4
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n==============================================")
print("          RESNET18 TEST RESULTS")
print("==============================================")

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)

print("\n----------------------------------------------")
print("Classification Report")
print("----------------------------------------------")

print(report)

print("----------------------------------------------")
print("Confusion Matrix")
print("----------------------------------------------")

print(cm)

print("==============================================")

