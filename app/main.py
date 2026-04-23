from fastapi import FastAPI

from app.routers import health, interactions, predict, symptoms

app = FastAPI(
    title="HMS AI Service",
    description=(
        "Internal AI/ML microservice for the Hospital Management System. "
        "Provides disease prediction from symptoms and drug interaction checking."
    ),
    version="0.2.0",
    docs_url="/docs",
)

# --- Health ---
app.include_router(health.router, tags=["Health"])

# --- Disease Prediction ---
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(symptoms.router, prefix="/symptoms", tags=["Prediction"])

# --- Drug Interactions ---
app.include_router(interactions.router, prefix="/interactions", tags=["Interactions"])
