from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any

router = APIRouter()


class PredictRequest(BaseModel):
    data: dict[str, Any]


class PredictResponse(BaseModel):
    prediction: str
    confidence: float


@router.post("", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Dummy prediction endpoint. Replace with real ML model later."""
    return PredictResponse(
        prediction="dummy_result",
        confidence=0.95,
    )
