"""Adapter that exposes the trained disease model through AgriMind's agent contract."""

from io import BytesIO
from threading import Lock

import torch
from PIL import Image, UnidentifiedImageError

from agents.disease_agent.src.dataset import CLASS_NAMES
from agents.disease_agent.src.inference import DEVICE, load_model
from agents.disease_agent.src.preprocessing import get_eval_transforms


class DiseaseAgent:
    """Runs image classification and hides PyTorch details from the orchestrator."""

    name = "disease_agent"
    model_version = "resnet18_best"

    def __init__(self) -> None:
        self._model = None
        self._model_lock = Lock()

    def _get_model(self):
        """Load the  model once per API process, safely under concurrent requests."""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    try:
                        self._model = load_model()
                    except FileNotFoundError as error:
                        raise RuntimeError(
                            "Disease model is unavailable. Expected "
                            "agents/disease_agent/models/resnet18_best.pth."
                        ) from error
        return self._model

    def analyze_image(self, image_bytes: bytes) -> dict:
        """Classify image bytes and return the common specialist-agent result shape."""
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError("The upload is not a readable image.") from error

        image_tensor = get_eval_transforms()(image).unsqueeze(0).to(DEVICE)
        model = self._get_model()
        with torch.no_grad():
            probabilities = torch.softmax(model(image_tensor), dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)

        confidence_value = float(confidence.item())
        warnings: list[str] = []
        if confidence_value < 0.70:
            warnings.append(
                "Low-confidence result. Upload a clear close-up of one affected leaf "
                "or seek agronomist confirmation."
            )

        return {
            "agent": self.name,
            "status": "success",
            "data": {
                "name": CLASS_NAMES[predicted_class.item()],
                "confidence": confidence_value,
                "model_version": self.model_version,
            },
            "warnings": warnings,
        }
