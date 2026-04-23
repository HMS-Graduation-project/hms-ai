"""Disease prediction endpoint.

Accepts a list of symptom identifiers and returns the top-3 most likely
diseases ranked by Jaccard similarity confidence score.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ..services.disease_predictor import ALL_SYMPTOMS, predict_diseases

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictionItem(BaseModel):
    """A single disease prediction."""

    disease: str = Field(..., examples=["Influenza"])
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Jaccard similarity score between 0 and 1",
        examples=[0.4286],
    )


class PredictRequest(BaseModel):
    """Request body for the /predict endpoint."""

    symptoms: list[str] = Field(
        ...,
        min_length=1,
        description="List of symptom identifiers",
        examples=[["fever", "cough", "fatigue"]],
    )


class PredictResponse(BaseModel):
    """Response body for the /predict endpoint."""

    predictions: list[PredictionItem]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=PredictResponse,
    summary="Predict diseases from symptoms",
    description=(
        "Given a list of symptom identifiers, returns up to 3 disease "
        "predictions ranked by confidence (Jaccard similarity). "
        "Use GET /symptoms to discover valid symptom identifiers."
    ),
)
async def predict(request: PredictRequest) -> PredictResponse:
    """Return top disease predictions for the given symptoms."""
    # Validate that at least one symptom is recognised
    known = set(ALL_SYMPTOMS)
    unknown = [s for s in request.symptoms if s not in known]
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Unknown symptom identifiers",
                "unknown_symptoms": unknown,
                "hint": "Use GET /symptoms to list valid identifiers",
            },
        )

    results = predict_diseases(request.symptoms, top_k=3)

    if not results:
        return PredictResponse(predictions=[])

    return PredictResponse(
        predictions=[
            PredictionItem(disease=r.disease, confidence=r.confidence)
            for r in results
        ],
    )
