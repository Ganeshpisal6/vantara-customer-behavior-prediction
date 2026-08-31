from fastapi import FastAPI
from api.routers import prediction

app = FastAPI(
    title="Vantara Customer Behavior Prediction API",
    description="API for customer churn prediction and customer intelligence",
    version="1.0.0"
)

app.include_router(prediction.router)


@app.get("/")
def root():
    return {
        "message": "Vantara Customer Behavior Prediction API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }