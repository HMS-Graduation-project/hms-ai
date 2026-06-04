"""
Grad-CAM for DenseNet121 pneumonia classifier.

Avoids DenseNet inplace ReLU issues by manually splitting the forward
pass and computing gradients on the feature map tensor directly.
"""

from __future__ import annotations

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


def _disable_inplace(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False


def load_model(checkpoint_path: Path | str | None = None) -> nn.Module:
    ckpt_path = Path(checkpoint_path) if checkpoint_path else BEST_MODEL_PATH
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _disable_inplace(model)
    model.to(DEVICE).eval()
    return model


def preprocess_image(path: Path | str) -> tuple[torch.Tensor, np.ndarray]:
    img = Image.open(path).convert("RGB")
    original = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tensor = transform(img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS))
    return tensor.unsqueeze(0).to(DEVICE), original


class GradCAM:
    """Grad-CAM that manually splits the DenseNet forward pass to avoid
    inplace-op gradient errors in DenseNet's dense blocks."""

    def __init__(self, model: nn.Module):
        self.model = model

    @torch.enable_grad()
    def generate(
        self, image_path: Path | str, target_class: int | None = None,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        self.model.eval()
        input_tensor, _ = preprocess_image(image_path)

        # Manual forward: features -> relu -> pool -> flatten -> classifier
        # Keep features as a leaf-like tensor that requires grad
        with torch.no_grad():
            features_raw = self.model.features(input_tensor)

        # Detach and re-attach to create a clean grad graph
        features = features_raw.detach().clone().requires_grad_(True)

        out = F.relu(features)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        output = self.model.classifier(out)

        probs = F.softmax(output, dim=1).detach().cpu().numpy()[0]
        pred_class = int(output.argmax(dim=1).item())

        if target_class is None:
            target_class = pred_class

        # Backward
        score = output[0, target_class]
        score.backward()

        # Grad-CAM: weight activations by gradient
        gradients = features.grad[0]   # [C, H, W]
        activations = features[0]      # [C, H, W]

        weights = gradients.mean(dim=(1, 2))  # [C]
        cam = (weights[:, None, None] * activations).sum(dim=0)
        cam = F.relu(cam).detach().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        cam_resized = np.array(Image.fromarray(cam.astype(np.float32)).resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS,
        ))

        return cam_resized, pred_class, probs

    def generate_overlay(
        self, image_path: Path | str, alpha: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.cm as cm

        heatmap, pred_class, probs = self.generate(image_path)
        _, original = preprocess_image(image_path)

        colormap = cm.jet(heatmap)[:, :, :3]
        heatmap_colored = (colormap * 255).astype(np.uint8)

        overlay = (original.astype(np.float32) * (1 - alpha) +
                   heatmap_colored.astype(np.float32) * alpha)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        return original, heatmap_colored, overlay, pred_class, probs
