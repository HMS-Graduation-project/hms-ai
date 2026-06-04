"""Pydantic schemas for pneumonia prediction API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PneumoniaHealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    modelLoaded: bool
    modelVersion: str = Field(..., examples=["pneumonia-densenet121-v2"])
    threshold: float = Field(..., examples=[0.94])
    device: str = Field(..., examples=["cuda"])


class PneumoniaPredictionResponse(BaseModel):
    prediction: str = Field(..., examples=["PNEUMONIA"])
    probability: float = Field(..., ge=0, le=1, examples=[0.9732])
    confidence: float = Field(..., ge=0, le=1, examples=[0.9732])
    threshold: float = Field(..., examples=[0.94])
    isPositive: bool = Field(..., examples=[True])
    modelVersion: str = Field(..., examples=["pneumonia-densenet121-v2"])
    device: str = Field(..., examples=["cuda"])
    clinicalNote: str = Field(
        ...,
        examples=["AI-assisted screening result. Not a final diagnosis."],
    )


class ExplainabilityInfo(BaseModel):
    type: str = Field(default="Grad-CAM")
    overlayImageBase64: str = Field(..., description="Base64-encoded overlay PNG")
    heatmapImageBase64: str = Field(..., description="Base64-encoded heatmap PNG")
    clinicalNote: str = Field(
        default="Heatmap shows regions influencing model prediction. "
        "Red/yellow = high influence. Blue = low influence.",
    )


class PneumoniaExplainResponse(PneumoniaPredictionResponse):
    explainability: ExplainabilityInfo
