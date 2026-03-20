from fastapi import FastAPI
from app.routers import health, predict

app = FastAPI(
    title="HMS AI Service",
    description="Internal AI/ML service for HMS",
    version="0.1.0",
    docs_url="/docs",
)

app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, prefix="/predict", tags=["Predict"])
