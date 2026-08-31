import pandas as pd
import numpy as np
from pathlib import Path


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "customer_features.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

OUTPUT_FILE = OUTPUT_DIR / "customer_model_features.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# LOAD CUSTOMER FEATURES
# ------------------------------------------------------------

print("Loading customer features...")

df = pd.read_csv(INPUT_FILE)

print(f"Customers loaded: {len(df):,}")


# ------------------------------------------------------------
# RFM SCORES
# ------------------------------------------------------------

print("Creating RFM scores...")


df["recency_score"] = pd.qcut(
    df["recency"],
    5,
    labels=[5, 4, 3, 2, 1],
    duplicates="drop"
).astype(int)


df["frequency_score"] = pd.qcut(
    df["frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)


df["monetary_score"] = pd.qcut(
    df["monetary"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)


# ------------------------------------------------------------
# RFM TOTAL SCORE
# ------------------------------------------------------------

df["rfm_score"] = (
    df["recency_score"]
    + df["frequency_score"]
    + df["monetary_score"]
)


# ------------------------------------------------------------
# CUSTOMER SEGMENT
# ------------------------------------------------------------

def assign_segment(score):

    if score >= 13:
        return "Champions"

    elif score >= 10:
        return "Loyal Customers"

    elif score >= 7:
        return "Potential Customers"

    else:
        return "At Risk"


df["customer_segment"] = df["rfm_score"].apply(assign_segment)


# ------------------------------------------------------------
# CHURN RISK
# ------------------------------------------------------------

recency_threshold = df["recency"].median()

df["churn_status"] = np.where(
    df["recency"] > recency_threshold,
    "High Risk",
    "Low Risk"
)


df["churn_probability"] = np.where(
    df["churn_status"] == "High Risk",
    0.75,
    0.15
)


# ------------------------------------------------------------
# CUSTOMER LIFETIME VALUE
# ------------------------------------------------------------

df["clv_score"] = (
    df["monetary"]
    * df["frequency"]
) / (
    df["recency"] + 1
)


# ------------------------------------------------------------
# CLV CATEGORY
# ------------------------------------------------------------

clv_75 = df["clv_score"].quantile(0.75)
clv_50 = df["clv_score"].quantile(0.50)


def assign_clv(value):

    if value >= clv_75:
        return "Very High Value"

    elif value >= clv_50:
        return "High Value"

    else:
        return "Standard Value"


df["clv_category"] = df["clv_score"].apply(assign_clv)


# ------------------------------------------------------------
# ACTION PRIORITY
# ------------------------------------------------------------

def assign_priority(row):

    if row["churn_status"] == "High Risk":
        return "High"

    elif row["clv_category"] in [
        "Very High Value",
        "High Value"
    ]:
        return "Medium"

    else:
        return "Low"


df["action_priority"] = df.apply(
    assign_priority,
    axis=1
)


# ------------------------------------------------------------
# BUSINESS RECOMMENDATION
# ------------------------------------------------------------

def recommendation(row):

    if row["churn_status"] == "High Risk":

        return (
            "Launch retention campaign and "
            "personalized offer"
        )

    elif row["clv_category"] == "Very High Value":

        return (
            "Provide premium rewards and "
            "exclusive offers"
        )

    elif row["customer_segment"] == "Potential Customers":

        return (
            "Use personalized promotions to "
            "increase customer engagement"
        )

    else:

        return (
            "Maintain regular engagement "
            "and monitor customer activity"
        )


df["business_recommendation"] = df.apply(
    recommendation,
    axis=1
)


# ------------------------------------------------------------
# SAVE FINAL MODEL DATASET
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

print("\nFeature engineering completed!")

print(
    f"Final customers: {len(df):,}"
)

print(
    f"Saved to: {OUTPUT_FILE}"
)

print("\nFinal columns:")

print(df.columns.tolist())

print("\nCustomer segments:")

print(
    df["customer_segment"]
    .value_counts()
)

print("\nChurn status:")

print(
    df["churn_status"]
    .value_counts()
)

