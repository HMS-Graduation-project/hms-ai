"""
Pneumonia detection API endpoints.

POST /predict -- chest X-ray classification
POST /explain -- classification + Grad-CAM heatmap
GET  /health  -- model status
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from app.schemas.pneumonia import (
    PneumoniaHealthResponse,
    PneumoniaPredictionResponse,
    PneumoniaExplainResponse,
)
from app.services import pneumonia_service

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}
MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


def _validate_upload(file: UploadFile, data: bytes) -> None:
    if not file.filename:
        raise HTTPException(400, "No file uploaded")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in {"jpg", "jpeg", "png"}:
        raise HTTPException(400, f"Unsupported file type '.{ext}'. Use JPG or PNG.")

    if len(data) > MAX_SIZE_BYTES:
        raise HTTPException(413, f"File too large ({len(data) // 1024} KB). Max 10 MB.")

    if len(data) < 100:
        raise HTTPException(400, "File is too small to be a valid image.")


@router.get(
    "/health",
    response_model=PneumoniaHealthResponse,
    summary="Pneumonia model health check",
    description="Returns model loading status, version, threshold, and device.",
)
async def health():
    try:
        return pneumonia_service.get_health()
    except Exception as e:
        raise HTTPException(500, f"Health check failed: {e}")


@router.post(
    "/predict",
    response_model=PneumoniaPredictionResponse,
    summary="Predict pneumonia from chest X-ray",
    description=(
        "Upload a chest X-ray image (JPG/PNG, max 10 MB). "
        "Returns NORMAL or PNEUMONIA prediction with probability and confidence. "
        "Uses optimized clinical threshold (0.94). "
        "AI-assisted screening only -- not a final diagnosis."
    ),
)
async def predict(file: UploadFile = File(..., description="Chest X-ray image (JPG/PNG)")):
    data = await file.read()
    _validate_upload(file, data)

    try:
        result = pneumonia_service.predict(data)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")

    return result


@router.post(
    "/explain",
    response_model=PneumoniaExplainResponse,
    summary="Predict pneumonia with Grad-CAM explanation",
    description=(
        "Upload a chest X-ray image. Returns prediction plus Grad-CAM "
        "heatmap and overlay as base64-encoded PNG images. "
        "Shows which regions influenced the model's decision."
    ),
)
async def explain(file: UploadFile = File(..., description="Chest X-ray image (JPG/PNG)")):
    data = await file.read()
    _validate_upload(file, data)

    try:
        result = pneumonia_service.explain(data)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, f"Explanation failed: {e}")

    return result
