from pathlib import Path

import pandas as pd
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Details",
    page_icon="👤",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = BASE_DIR / "data" / "processed" / "customer_model_features.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)

    df = df.rename(columns={
        "customer_id": "Customer ID",
        "customer_segment": "Customer_Segment",
        "churn_status": "Churn_Status",
        "churn_probability": "Churn_Probability",
        "clv_score": "CLV_Score",
        "clv_category": "CLV_Category",
        "action_priority": "Action_Priority",
        "business_recommendation": "Business_Recommendation"
    })

    return df


df = load_data()


# ============================================================
# TITLE
# ============================================================

st.title("Customer Details")
st.write("View detailed information about an individual customer.")


# ============================================================
# CUSTOMER SELECTION
# ============================================================

customer_ids = sorted(df["Customer ID"].dropna().unique())

selected_customer = st.selectbox(
    "Select Customer ID",
    customer_ids
)


# ============================================================
# SELECT CUSTOMER
# ============================================================

customer = df[df["Customer ID"] == selected_customer]


if not customer.empty:

    row = customer.iloc[0]


    # ========================================================
    # CUSTOMER OVERVIEW
    # ========================================================

    st.subheader("Customer Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Customer ID",
            str(int(row["Customer ID"]))
        )

    with col2:
        st.metric(
            "Customer Segment",
            row["Customer_Segment"]
        )

    with col3:
        st.metric(
            "Churn Status",
            row["Churn_Status"]
        )


    # ========================================================
    # CUSTOMER INFORMATION
    # ========================================================

    st.subheader("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(f"**Recency:** {row['recency']}")
        st.write(f"**Frequency:** {row['frequency']}")

    with col2:
        st.write(f"**Monetary:** {row['monetary']:,.2f}")
        st.write(f"**CLV Score:** {row['CLV_Score']:,.4f}")

    with col3:
        st.write(f"**CLV Category:** {row['CLV_Category']}")
        st.write(f"**Action Priority:** {row['Action_Priority']}")


    # ========================================================
    # BUSINESS RECOMMENDATION
    # ========================================================

    st.subheader("Business Recommendation")

    st.info(
        row["Business_Recommendation"]
    )


    # ========================================================
    # CHURN PROBABILITY
    # ========================================================

    st.subheader("Churn Prediction")

    probability = row["Churn_Probability"]

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )


    # ========================================================
    # MODEL FEATURES
    # ========================================================

    st.subheader("Customer Model Features")

    feature_columns = [
        "recency",
        "frequency",
        "monetary",
        "total_quantity",
        "average_order_value",
        "unique_products",
        "recency_score",
        "frequency_score",
        "monetary_score",
        "rfm_score"
    ]

    available_features = [
        col for col in feature_columns
        if col in df.columns
    ]

    st.dataframe(
        customer[available_features],
        use_container_width=True
    )

else:

    st.warning("Customer not found.")