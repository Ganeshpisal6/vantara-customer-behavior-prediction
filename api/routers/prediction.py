import io
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from api.schemas.customer import CustomerFeatures

router = APIRouter(
    prefix="/api",
    tags=["Customer Prediction"]
)

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    BASE_DIR
    / "models_artifacts"
    / "xgboost.pkl"
)

model = joblib.load(MODEL_FILE)


# Features actually used by the trained XGBoost model
MODEL_FEATURES = [
    "frequency",
    "monetary",
    "total_quantity",
    "average_order_value",
    "unique_products",
    "frequency_score",
    "monetary_score",
    "rfm_score",
    "clv_score",
]


@router.get("/test")
def test_prediction_api():
    return {
        "message": "Prediction API is working"
    }


@router.post("/predict")
def predict_churn(data: CustomerFeatures):

    input_data = pd.DataFrame(
        [data.model_dump()]
    )

    input_data = input_data[
        MODEL_FEATURES
    ]

    prediction = model.predict(
        input_data
    )[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    churn_status = (
        "High Risk"
        if prediction == 1
        else "Low Risk"
    )

    return {
        "churn_prediction": int(prediction),
        "churn_status": churn_status,
        "churn_probability": round(
            float(probability),
            3
        )
    }


@router.post("/predict/batch")
async def predict_batch(
    file: UploadFile = File(...)  # noqa: B008
):

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported."
        )

    contents = await file.read()

    try:
        df = pd.read_csv(
            io.BytesIO(contents)
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=f"Unable to read CSV file: {exc}"
        )

    missing_columns = [
        column
        for column in MODEL_FEATURES
        if column not in df.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required columns",
                "columns": missing_columns
            }
        )

    X = df[MODEL_FEATURES]

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    result = df.copy()

    result["churn_prediction"] = (
        predictions.astype(int)
    )

    result["churn_status"] = (
        result["churn_prediction"]
        .map({
            0: "Low Risk",
            1: "High Risk"
        })
    )

    result["churn_probability"] = probabilities

    return {
        "customers_scored": len(result),
        "predictions": result.to_dict(
            orient="records"
        )
    }


@router.get("/metadata")
def api_metadata():

    return {
        "api_name": "Vantara Customer Behavior Prediction API",
        "version": "1.0.0",
        "model": "XGBoost Classifier",
        "prediction_type": "Customer Churn Prediction",
        "supported_endpoints": [
            "/api/test",
            "/api/metadata",
            "/api/predict",
            "/api/predict/batch"
        ],
        "required_features": MODEL_FEATURES
    }