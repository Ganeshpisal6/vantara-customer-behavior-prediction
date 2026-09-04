from pathlib import Path

import joblib
import pandas as pd
import shap
import streamlit as st

# ============================================================
# VANTARA - CHURN PREDICTION
# ============================================================

st.set_page_config(
    page_title="Vantara - Churn Prediction",
    page_icon="🔮",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "customer_model_features.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "models_artifacts"
    / "churn_model.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


df = load_data()
model = load_model()


# ============================================================
# TITLE
# ============================================================

st.title("Vantara Churn Prediction")

st.write(
    "Predict customer churn risk using the trained Random Forest model."
)


# ============================================================
# CUSTOMER SELECTION
# ============================================================

st.subheader("Select Customer")

customer_ids = df["customer_id"].astype(int).astype(str).tolist()

selected_id = st.selectbox(
    "Customer ID",
    customer_ids
)


# ============================================================
# GET CUSTOMER
# ============================================================

customer = df[
    df["customer_id"].astype(int).astype(str) == selected_id
].iloc[0]


# ============================================================
# MODEL FEATURES
# ============================================================

features = [
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


X = customer[features].to_frame().T


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Customer ID",
        selected_id
    )

with col2:
    st.metric(
        "Current Churn Status",
        customer["churn_status"]
    )

with col3:
    st.metric(
        "Current CLV Category",
        customer["clv_category"]
    )


# ============================================================
# PREDICTION
# ============================================================

if st.button("Predict Churn Risk"):

    prediction = model.predict(X)[0]

    probability = model.predict_proba(X)[0][1]

    if prediction == 1:
        churn_status = "High Risk"
    else:
        churn_status = "Low Risk"

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Churn Status",
            churn_status
        )

    with col2:
        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )


    # ========================================================
    # RECOMMENDATION
    # ========================================================

    if prediction == 1:

        st.warning(
            "This customer is predicted to be at high risk of churn. "
            "Consider retention offers, personalized communication "
            "and follow-up campaigns."
        )

    else:

        st.success(
            "This customer is predicted to be at low risk of churn. "
            "Continue engagement and loyalty activities."
        )


# ============================================================
# MODEL INPUT DETAILS
# ============================================================

st.subheader("Model Input Features")

display_features = customer[features].to_frame().T

st.dataframe(
    display_features,
    use_container_width=True
)


# ============================================================
# SHAP EXPLANATION
# ============================================================

st.subheader("Why is this customer at this churn risk?")

try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        values = shap_values[1][0]

    elif len(shap_values.shape) == 3:
        # Format: samples x features x classes
        values = shap_values[0, :, 1]

    else:
        values = shap_values[0]

    explanation_df = pd.DataFrame({
        "Feature": features,
        "Value": X.iloc[0].values,
        "SHAP Impact": values
    })

    explanation_df["Impact"] = explanation_df["SHAP Impact"].apply(
        lambda x: "Increases Churn Risk"
        if x > 0
        else "Decreases Churn Risk"
    )

    explanation_df["Absolute Impact"] = (
        explanation_df["SHAP Impact"].abs()
    )

    explanation_df = explanation_df.sort_values(
        by="Absolute Impact",
        ascending=False
    )

    explanation_df = explanation_df[
        ["Feature", "Value", "SHAP Impact", "Impact"]
    ]

    st.dataframe(
        explanation_df,
        use_container_width=True,
        hide_index=True
    )

except Exception as e:  # noqa: BLE001
    st.warning(f"SHAP explanation error: {e}") 