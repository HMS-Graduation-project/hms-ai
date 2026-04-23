"""Symptoms listing endpoint.

Provides the complete catalogue of recognised symptom identifiers so that
consumers (e.g. the frontend) can build a multi-select picker.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..services.disease_predictor import ALL_SYMPTOMS

router = APIRouter()


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class SymptomsResponse(BaseModel):
    """Response body for the /symptoms endpoint."""

    symptoms: list[str] = Field(
        ...,
        description="Sorted list of all recognised symptom identifiers",
    )
    total: int = Field(..., description="Total number of symptoms")


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=SymptomsResponse,
    summary="List all recognised symptoms",
    description=(
        "Returns the full catalogue of symptom identifiers that can be "
        "passed to POST /predict. Useful for building a multi-select UI."
    ),
)
async def list_symptoms() -> SymptomsResponse:
    """Return every symptom known to the prediction engine."""
    return SymptomsResponse(
        symptoms=ALL_SYMPTOMS,
        total=len(ALL_SYMPTOMS),
    )
