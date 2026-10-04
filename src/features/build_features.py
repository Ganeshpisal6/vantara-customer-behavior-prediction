from pathlib import Path

import numpy as np
import pandas as pd

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

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "customer_model_features.csv"
)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("Loading customer features...")

df = pd.read_csv(INPUT_FILE)

print(f"Customers loaded: {len(df):,}")


# ------------------------------------------------------------
# DATE CONVERSION
# ------------------------------------------------------------

df["last_purchase_date"] = pd.to_datetime(
    df["last_purchase_date"],
    errors="coerce"
)

df = df.dropna(
    subset=["last_purchase_date"]
)


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


df["customer_segment"] = (
    df["rfm_score"].apply(assign_segment)
)


# ------------------------------------------------------------
# 90-DAY CHURN TARGET
# ------------------------------------------------------------

print("Creating 90-day churn target...")

# The dataset ends at this date.
dataset_end = df["last_purchase_date"].max()

# A customer is considered churned if they have not purchased
# during the following 90 days.
#
# IMPORTANT:
# Customers whose last purchase is too close to the dataset
# end cannot be reliably labelled because their complete
# 90-day future window is not observable.

cutoff_date = (
    dataset_end - pd.Timedelta(days=90)
)

df["churn_target"] = (
    df["last_purchase_date"] <= cutoff_date
).astype(int)


# ------------------------------------------------------------
# CHURN STATUS
# ------------------------------------------------------------

df["churn_status"] = np.where(
    df["churn_target"] == 1,
    "High Risk",
    "Low Risk"
)


# ------------------------------------------------------------
# CHURN PROBABILITY
# ------------------------------------------------------------

# Placeholder probability for dashboard compatibility.
# The trained ML model will provide the actual probability.

df["churn_probability"] = np.where(
    df["churn_target"] == 1,
    0.75,
    0.15
)


# ------------------------------------------------------------
# CUSTOMER LIFETIME VALUE
# ------------------------------------------------------------

df["clv_score"] = (
    df["monetary"] * df["frequency"]
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


df["clv_category"] = (
    df["clv_score"].apply(assign_clv)
)


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


df["action_priority"] = (
    df.apply(assign_priority, axis=1)
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


df["business_recommendation"] = (
    df.apply(recommendation, axis=1)
)


# ------------------------------------------------------------
# SAVE
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

print("\nChurn target distribution:")
print(
    df["churn_target"].value_counts()
)

print("\nChurn status distribution:")
print(
    df["churn_status"].value_counts()
)