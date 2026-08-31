import io
import joblib
import pandas as pd

from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException

from api.schemas.customer import CustomerFeatures


# ------------------------------------------------------------
# ROUTER
# ------------------------------------------------------------

router = APIRouter(
    prefix="/api",
    tags=["Customer Prediction"]
)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    BASE_DIR
    / "models_artifacts"
    / "churn_model.pkl"
)


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

model = joblib.load(MODEL_FILE)


# ------------------------------------------------------------
# TEST ENDPOINT
# ------------------------------------------------------------

@router.get("/test")
def test_prediction_api():

    return {
        "message": "Prediction API is working"
    }


# ------------------------------------------------------------
# SINGLE CUSTOMER PREDICTION
# ------------------------------------------------------------

@router.post("/predict")
def predict_churn(data: CustomerFeatures):

    input_data = pd.DataFrame(
        [data.model_dump()]
    )

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


# ------------------------------------------------------------
# BATCH CUSTOMER PREDICTION
# ------------------------------------------------------------

@router.post("/predict/batch")
async def predict_batch(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # CHECK FILE TYPE
    # --------------------------------------------------------

    if not file.filename.lower().endswith(".csv"):

        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported."
        )


    # --------------------------------------------------------
    # READ FILE
    # --------------------------------------------------------

    contents = await file.read()

    try:

        df = pd.read_csv(
            io.BytesIO(contents)
        )

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=f"Unable to read CSV file: {exc}"
        )


    # --------------------------------------------------------
    # REQUIRED FEATURES
    # --------------------------------------------------------

    required_features = [

        "recency",
        "frequency",
        "monetary",
        "total_quantity",
        "average_order_value",
        "unique_products",
        "recency_score",
        "frequency_score",
        "monetary_score",
        "rfm_score",
        "clv_score"

    ]


    # --------------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # --------------------------------------------------------

    missing_columns = [

        column
        for column in required_features
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


    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

    X = df[
        required_features
    ]


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    predictions = model.predict(
        X
    )

    probabilities = model.predict_proba(
        X
    )[:, 1]


    # --------------------------------------------------------
    # CREATE RESULT
    # --------------------------------------------------------

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


    result["churn_probability"] = (
        probabilities
    )


    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return {

        "customers_scored": len(result),

        "predictions": result.to_dict(
            orient="records"
        )

    }
    