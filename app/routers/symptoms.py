"""Symptoms listing endpoint.

Provides the complete catalogue of recognised symptoms so that consumers
(e.g. the frontend) can build a multi-select picker. Each entry exposes both
the canonical ``id`` (the exact value accepted by POST /predict) and a
human-readable ``label`` for display.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..services.disease_predictor import ALL_SYMPTOMS

router = APIRouter()


def humanize(symptom_id: str) -> str:
    """Turn a snake_case symptom id into a human-readable label.

    e.g. ``"loss_of_taste"`` -> ``"Loss of taste"``.
    """
    return symptom_id.replace("_", " ").strip().capitalize()


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class SymptomItem(BaseModel):
    """A single catalogue entry: canonical id + display label."""

    id: str = Field(..., description="Canonical symptom identifier")
    label: str = Field(..., description="Human-readable symptom name")


class SymptomsResponse(BaseModel):
    """Response body for the /symptoms endpoint."""

    symptoms: list[SymptomItem] = Field(
        ...,
        description="Sorted catalogue of recognised symptoms (id + label)",
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
        "Returns the full catalogue of symptoms (id + label) that can be "
        "passed to POST /predict. Submit the `id` values; show the `label`."
    ),
)
async def list_symptoms() -> SymptomsResponse:
    """Return every symptom known to the prediction engine."""
    items = [SymptomItem(id=s, label=humanize(s)) for s in ALL_SYMPTOMS]
    return SymptomsResponse(symptoms=items, total=len(items))
