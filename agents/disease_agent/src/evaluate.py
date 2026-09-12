import torch
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)
from agents.disease_agent.src.dataset import create_dataloaders
from agents.disease_agent.src.model import BaselineCNN


MODEL_PATH = "agents/disease_agent/models/baseline_cnn_best.pth"

CLASS_NAMES = [
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

_, _, test_loader = create_dataloaders()

model = BaselineCNN(num_classes=5).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())


# Accuracy
accuracy = accuracy_score(all_labels, all_predictions)

print("\n" + "=" * 60)
print("BASELINE CNN EVALUATION")
print("=" * 60)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Classification report
print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        digits=4,
    )
)

# Confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

print("Confusion Matrix:")
print(cm)

print("\nClass order:")
for i, class_name in enumerate(CLASS_NAMES):
    print(f"{i}: {class_name}")

print("=" * 60)