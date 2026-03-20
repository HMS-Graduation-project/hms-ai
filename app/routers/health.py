from fastapi import APIRouter
from datetime import datetime

router = APIRouter()


@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "hms-ai",
        "timestamp": datetime.utcnow().isoformat(),
    }
