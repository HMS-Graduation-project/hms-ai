"""Drug interaction checking endpoints.

Accepts a list of medication names and returns all known pairwise
interactions found in the curated database.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ..services.interaction_checker import check_interactions, get_all_medications

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class InteractionItem(BaseModel):
    """A single drug-drug interaction."""

    drug1: str = Field(..., examples=["Warfarin"])
    drug2: str = Field(..., examples=["Aspirin"])
    severity: str = Field(
        ...,
        description="Severity level: HIGH, MODERATE, or LOW",
        examples=["HIGH"],
    )
    description: str = Field(
        ...,
        examples=[
            "Increased risk of bleeding. Combined use significantly "
            "raises hemorrhagic risk."
        ],
    )


class InteractionCheckRequest(BaseModel):
    """Request body for the /interactions endpoint."""

    medications: list[str] = Field(
        ...,
        min_length=2,
        description="List of medication names to check (minimum 2)",
        examples=[["Warfarin", "Aspirin", "Metformin"]],
    )


class InteractionCheckResponse(BaseModel):
    """Response body for the /interactions endpoint."""

    interactions: list[InteractionItem]
    total: int = Field(..., description="Number of interactions found")


class MedicationsResponse(BaseModel):
    """Response body for the /medications endpoint."""

    medications: list[str] = Field(
        ...,
        description="Sorted list of all medication names in the database",
    )
    total: int = Field(..., description="Total number of medications")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=InteractionCheckResponse,
    summary="Check drug interactions",
    description=(
        "Given a list of medication names, checks all pairwise combinations "
        "for known interactions. Results are sorted by severity (HIGH first). "
        "Use GET /medications to discover valid medication names."
    ),
)
async def check_drug_interactions(
    request: InteractionCheckRequest,
) -> InteractionCheckResponse:
    """Check all pairwise drug interactions for the given medications."""
    known = {m.lower() for m in get_all_medications()}
    unknown = [m for m in request.medications if m.lower() not in known]
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Unknown medication names",
                "unknown_medications": unknown,
                "hint": "Use GET /medications to list valid names",
            },
        )

    results = check_interactions(request.medications)

    return InteractionCheckResponse(
        interactions=[
            InteractionItem(
                drug1=r.drug1,
                drug2=r.drug2,
                severity=r.severity,
                description=r.description,
            )
            for r in results
        ],
        total=len(results),
    )


@router.get(
    "/medications",
    response_model=MedicationsResponse,
    summary="List all known medications",
    description=(
        "Returns every medication name present in the interaction database. "
        "Useful for building an autocomplete or multi-select UI."
    ),
)
async def list_medications() -> MedicationsResponse:
    """Return all medication names from the interaction database."""
    meds = get_all_medications()
    return MedicationsResponse(
        medications=meds,
        total=len(meds),
    )
