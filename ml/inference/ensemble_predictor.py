"""
Multi-model ensemble predictor for pneumonia screening.

Runs DenseNet121, EfficientNet-B0, and ResNet50 in sequence,
then combines their probabilities using a weighted average.
Models are loaded lazily and cached as singletons.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from ml.training.config import (
    CLASS_NAMES, DEVICE, IMAGE_SIZE,
    IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES,
)
from ml.inference.pneumonia_predictor import CLINICAL_THRESHOLD

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

# ── Model registry ──────────────────────────────────────────────────

MODEL_CONFIGS = {
    "DenseNet121": {
        "arch": "densenet121",
        "checkpoint": CHECKPOINT_DIR / "pneumonia_densenet121_best.pt",
        "version": "pneumonia-densenet121-v2",
    },
    "EfficientNet-B0": {
        "arch": "efficientnet_b0",
        "checkpoint": CHECKPOINT_DIR / "pneumonia_efficientnet_b0_best.pt",
        "version": "pneumonia-efficientnet-b0-v1",
    },
    "ResNet50": {
        "arch": "resnet50",
        "checkpoint": CHECKPOINT_DIR / "pneumonia_resnet50_best.pt",
        "version": "pneumonia-resnet50-v1",
    },
}

# Weights derived from relative model performance (F1 / AUC ranking)
DEFAULT_WEIGHTS = {
    "DenseNet121": 0.40,
    "EfficientNet-B0": 0.35,
    "ResNet50": 0.25,
}

ENSEMBLE_VERSION = "pneumonia-ensemble-v1"


def _build_model(arch: str) -> nn.Module:
    if arch == "densenet121":
        m = models.densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, NUM_CLASSES)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    elif arch == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return m


class EnsemblePredictor:
    """Lazy-loading multi-model ensemble. Each model loads on first use."""

    _instance: EnsemblePredictor | None = None

    def __init__(self, threshold: float = CLINICAL_THRESHOLD):
        self._models: dict[str, nn.Module] = {}
        self._device = DEVICE
        self._threshold = threshold
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    @classmethod
    def get_instance(cls) -> EnsemblePredictor:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self, name: str) -> nn.Module:
        if name in self._models:
            return self._models[name]

        cfg = MODEL_CONFIGS[name]
        ckpt_path = cfg["checkpoint"]
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found for {name}: {ckpt_path}"
            )

        model = _build_model(cfg["arch"])
        ckpt = torch.load(ckpt_path, map_location=self._device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self._device).eval()
        self._models[name] = model
        logger.info("Loaded %s (device=%s)", name, self._device)
        return model

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS,
        )
        return self._transform(image).unsqueeze(0).to(self._device)

    @torch.no_grad()
    def _predict_single(self, name: str, input_tensor: torch.Tensor) -> dict:
        model = self._load_model(name)
        cfg = MODEL_CONFIGS[name]

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
            "modelName": name,
            "modelVersion": cfg["version"],
            "prediction": CLASS_NAMES[predicted_idx],
            "probability": round(prob_pneumonia, 4),
            "confidence": round(float(probs[predicted_idx]), 4),
            "threshold": self._threshold,
            "isPositive": is_positive,
            "device": str(self._device),
        }

    def predict_ensemble(self, image_bytes: bytes) -> dict:
        """Run all 3 models and produce a weighted-average ensemble result."""
        image = Image.open(io.BytesIO(image_bytes))
        input_tensor = self._preprocess(image)

        # Run each model
        model_results = []
        for name in MODEL_CONFIGS:
            result = self._predict_single(name, input_tensor)
            model_results.append(result)

        # Weighted average
        weights = DEFAULT_WEIGHTS
        final_prob = sum(
            r["probability"] * weights[r["modelName"]]
            for r in model_results
        )
        final_prob = round(final_prob, 4)

        # Final decision using threshold
        is_positive = final_prob >= self._threshold
        predicted_idx = 1 if is_positive else 0
        final_prediction = CLASS_NAMES[predicted_idx]
        final_confidence = round(final_prob if is_positive else 1.0 - final_prob, 4)

        # Risk level
        if final_prob >= self._threshold:
            risk_level = "HIGH"
        elif final_prob >= 0.70:
            risk_level = "ELEVATED"
        elif final_prob >= 0.30:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        # Model agreement
        predictions = [r["prediction"] for r in model_results]
        pneumonia_count = predictions.count("PNEUMONIA")
        normal_count = predictions.count("NORMAL")
        max_agree = max(pneumonia_count, normal_count)

        if max_agree == 3:
            agreement = "STRONG"
            agreement_score = 1.0
        elif max_agree == 2:
            agreement = "MODERATE"
            agreement_score = 0.67
        else:
            agreement = "LOW"
            agreement_score = 0.33

        return {
            "prediction": final_prediction,
            "probability": final_prob,
            "confidence": final_confidence,
            "threshold": self._threshold,
            "isPositive": is_positive,
            "riskLevel": risk_level,
            "modelVersion": ENSEMBLE_VERSION,
            "device": str(self._device),
            "clinicalNote": (
                "AI-assisted ensemble screening result. "
                "Not a final diagnosis. Physician review is required."
            ),
            "ensemble": {
                "method": "WEIGHTED_AVERAGE",
                "weights": weights,
                "modelAgreement": agreement,
                "agreementScore": agreement_score,
                "models": model_results,
            },
        }
