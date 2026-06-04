"""
Pneumonia Inference Predictor.

Lazy-loading wrapper around the trained DenseNet121 checkpoint.
Accepts an image path or raw bytes and returns a prediction dict.

Usage:
    from ml.inference.pneumonia_predictor import PneumoniaPredictor

    predictor = PneumoniaPredictor()
    result = predictor.predict("path/to/xray.jpg")
    print(result)
    # {"prediction": "PNEUMONIA", "confidence": 0.95, "probability": 0.95,
    #  "probabilities": {"NORMAL": 0.05, "PNEUMONIA": 0.95},
    #  "modelVersion": "pneumonia-densenet121-v1"}
"""

from __future__ import annotations

import io
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from ml.training.config import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    CROP_SIZE,
    DEVICE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_VERSION,
    NUM_CLASSES,
)


class PneumoniaPredictor:
    """Lazy-loading inference wrapper for the pneumonia DenseNet121 model."""

    def __init__(self, checkpoint_path: Path | str | None = None):
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path else BEST_MODEL_PATH
        self._model: nn.Module | None = None
        self._device = DEVICE
        self._transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def _load_model(self) -> nn.Module:
        """Load the model from checkpoint on first use."""
        if self._model is not None:
            return self._model

        if not self._checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self._checkpoint_path}\n"
                "Train the model first: python -m ml.training.train_pneumonia"
            )

        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)

        checkpoint = torch.load(
            self._checkpoint_path, map_location=self._device, weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self._device)
        model.eval()

        self._model = model
        return model

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Apply transforms and add batch dimension."""
        # Chest X-rays are grayscale; DenseNet121 expects 3-channel RGB
        image = image.convert("RGB")
        tensor = self._transform(image)
        return tensor.unsqueeze(0).to(self._device)

    def predict(self, image_path: str | Path) -> dict:
        """Predict from an image file path.

        Returns:
            dict with prediction, confidence, probability, probabilities, modelVersion
        """
        image = Image.open(image_path)
        return self._predict_image(image)

    def predict_from_bytes(self, image_bytes: bytes) -> dict:
        """Predict from raw image bytes (for API upload integration).

        Returns:
            dict with prediction, confidence, probability, probabilities, modelVersion
        """
        image = Image.open(io.BytesIO(image_bytes))
        return self._predict_image(image)

    @torch.no_grad()
    def _predict_image(self, image: Image.Image) -> dict:
        """Core prediction logic."""
        model = self._load_model()
        input_tensor = self._preprocess(image)

        if self._device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(input_tensor)
        else:
            logits = model(input_tensor)

        probs = F.softmax(logits, dim=1).squeeze(0)
        predicted_idx = probs.argmax().item()
        confidence = probs[predicted_idx].item()

        return {
            "prediction": CLASS_NAMES[predicted_idx],
            "confidence": round(confidence, 4),
            "probability": round(confidence, 4),
            "probabilities": {
                name: round(probs[i].item(), 4) for i, name in enumerate(CLASS_NAMES)
            },
            "modelVersion": MODEL_VERSION,
        }
