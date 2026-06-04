"""
Pneumonia Inference Predictor.

Lazy-loading singleton wrapper around the trained DenseNet121 checkpoint.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from ml.training.config import (
    BEST_MODEL_PATH, CLASS_NAMES, DEVICE, IMAGE_SIZE,
    IMAGENET_MEAN, IMAGENET_STD, MODEL_VERSION, NUM_CLASSES,
)

CLINICAL_THRESHOLD = 0.94


class PneumoniaPredictor:
    """Lazy-loading inference wrapper. Model loads on first prediction."""

    _instance: PneumoniaPredictor | None = None

    def __init__(self, checkpoint_path: Path | str | None = None,
                 threshold: float = CLINICAL_THRESHOLD):
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path else BEST_MODEL_PATH
        self._model: nn.Module | None = None
        self._device = DEVICE
        self._threshold = threshold
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    @classmethod
    def get_instance(cls) -> PneumoniaPredictor:
        """Singleton access -- reuses model across requests."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load_model(self) -> nn.Module:
        if self._model is not None:
            return self._model

        if not self._checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self._checkpoint_path}\n"
                "Train first: python -m ml.training.train_pneumonia"
            )

        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
        ckpt = torch.load(self._checkpoint_path, map_location=self._device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self._device).eval()

        self._model = model
        return model

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        return self._transform(image).unsqueeze(0).to(self._device)

    @torch.no_grad()
    def _predict_image(self, image: Image.Image) -> dict:
        model = self._load_model()
        input_tensor = self._preprocess(image)

        if self._device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(input_tensor)
        else:
            logits = model(input_tensor)

        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        prob_pneumonia = float(probs[1])
        is_positive = prob_pneumonia >= self._threshold
        predicted_idx = 1 if is_positive else 0

        return {
            "prediction": CLASS_NAMES[predicted_idx],
            "probability": round(prob_pneumonia, 4),
            "confidence": round(float(probs[predicted_idx]), 4),
            "threshold": self._threshold,
            "isPositive": is_positive,
            "probabilities": {name: round(float(probs[i]), 4) for i, name in enumerate(CLASS_NAMES)},
            "modelVersion": MODEL_VERSION,
            "device": str(self._device),
            "clinicalNote": "AI-assisted screening result. Not a final diagnosis.",
        }

    def predict(self, image_path: str | Path) -> dict:
        return self._predict_image(Image.open(image_path))

    def predict_from_bytes(self, data: bytes) -> dict:
        return self._predict_image(Image.open(io.BytesIO(data)))
