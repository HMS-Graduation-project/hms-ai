"""
Pneumonia prediction service.

Wraps the ML predictor and Grad-CAM for use by the FastAPI router.
Model loads lazily on first call and is reused across requests.
"""

from __future__ import annotations

import base64
import io
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy imports -- torch may not be installed in all environments
_predictor = None
_gradcam = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        from ml.inference.pneumonia_predictor import PneumoniaPredictor
        _predictor = PneumoniaPredictor.get_instance()
        logger.info("Pneumonia predictor initialized (device=%s)", _predictor._device)
    return _predictor


def _get_gradcam():
    global _gradcam
    if _gradcam is None:
        from ml.xai.gradcam import GradCAM, load_model
        model = load_model()
        _gradcam = GradCAM(model)
        logger.info("Grad-CAM initialized")
    return _gradcam


def is_model_loaded() -> bool:
    return _predictor is not None and _predictor.is_loaded


def get_health() -> dict:
    from ml.training.config import MODEL_VERSION, DEVICE
    from ml.inference.pneumonia_predictor import CLINICAL_THRESHOLD
    predictor = _get_predictor()
    return {
        "status": "ok",
        "modelLoaded": predictor.is_loaded,
        "modelVersion": MODEL_VERSION,
        "threshold": CLINICAL_THRESHOLD,
        "device": str(DEVICE),
    }


def predict(file_bytes: bytes) -> dict:
    """Run pneumonia prediction on uploaded image bytes."""
    # Validate image
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
    except Exception:
        raise ValueError("Uploaded file is not a valid image")

    predictor = _get_predictor()
    return predictor.predict_from_bytes(file_bytes)


def explain(file_bytes: bytes) -> dict:
    """Run prediction + Grad-CAM explanation."""
    import tempfile
    from pathlib import Path

    # Validate
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
    except Exception:
        raise ValueError("Uploaded file is not a valid image")

    # Prediction
    predictor = _get_predictor()
    result = predictor.predict_from_bytes(file_bytes)

    # Grad-CAM needs a file path -- write to temp
    cam = _get_gradcam()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        original, heatmap, overlay, pred_class, probs = cam.generate_overlay(tmp_path)

        # Encode as base64 PNG
        def _to_base64(arr: np.ndarray) -> str:
            img = Image.fromarray(arr)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii")

        result["explainability"] = {
            "type": "Grad-CAM",
            "overlayImageBase64": _to_base64(overlay),
            "heatmapImageBase64": _to_base64(heatmap),
            "clinicalNote": (
                "Heatmap shows regions influencing model prediction. "
                "Red/yellow = high influence. Blue = low influence."
            ),
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return result
