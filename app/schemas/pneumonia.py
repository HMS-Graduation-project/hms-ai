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


# ── Ensemble schemas ─────────────────────────────────────────────────


class SingleModelResult(BaseModel):
    modelName: str = Field(..., examples=["DenseNet121"])
    modelVersion: str = Field(..., examples=["pneumonia-densenet121-v2"])
    prediction: str = Field(..., examples=["PNEUMONIA"])
    probability: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    threshold: float
    isPositive: bool
    device: str


class EnsembleInfo(BaseModel):
    method: str = Field(default="WEIGHTED_AVERAGE")
    weights: dict[str, float] = Field(
        ...,
        examples=[{"DenseNet121": 0.40, "EfficientNet-B0": 0.35, "ResNet50": 0.25}],
    )
    modelAgreement: str = Field(..., examples=["STRONG"])
    agreementScore: float = Field(..., ge=0, le=1, examples=[1.0])
    models: list[SingleModelResult]


class EnsembleExplainabilityInfo(BaseModel):
    type: str = Field(default="Grad-CAM")
    sourceModel: str = Field(default="DenseNet121")
    overlayImageBase64: str
    heatmapImageBase64: str
    clinicalNote: str


class PneumoniaEnsembleResponse(BaseModel):
    prediction: str = Field(..., examples=["NORMAL"])
    probability: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    threshold: float = Field(..., examples=[0.94])
    isPositive: bool
    riskLevel: str = Field(..., examples=["ELEVATED"])
    modelVersion: str = Field(default="pneumonia-ensemble-v1")
    device: str
    clinicalNote: str
    ensemble: EnsembleInfo
    explainability: EnsembleExplainabilityInfo | None = None
