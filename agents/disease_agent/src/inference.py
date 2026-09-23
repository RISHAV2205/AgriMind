import torch
from PIL import Image
from torchvision import models
import torch
import torch.nn as nn

from agents.disease_agent.src.dataset import CLASS_NAMES
from agents.disease_agent.src.preprocessing import get_eval_transforms


MODEL_PATH = (
    "agents/disease_agent/models/resnet18_best.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)
    model.eval()

    return model


def predict(image_path):
    model = load_model()

    image = Image.open(image_path).convert("RGB")

    transform = get_eval_transforms()

    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = predicted_class.item()
    confidence_value = confidence.item()

    disease = CLASS_NAMES[predicted_index]

    return {
        "disease": disease,
        "confidence": confidence_value
    }


if __name__ == "__main__":

    result = predict(
    r"D:\AgriMind\agents\disease_agent\data\test\tomato-early-blight-11-768x510.jpg")

    print(result)