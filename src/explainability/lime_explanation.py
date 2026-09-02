from pathlib import Path

import joblib
import pandas as pd

from lime.lime_tabular import LimeTabularExplainer


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

OUTPUT_DIR = (
    BASE_DIR
    / "models_artifacts"
    / "lime_outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("VANTARA - LIME EXPLAINABILITY")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Customers loaded: {len(df):,}")


# ============================================================
# FEATURES
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

X = df[features].copy()


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading churn model...")

model = joblib.load(MODEL_FILE)

print(f"Model loaded: {type(model).__name__}")


# ============================================================
# CREATE PREDICTION FUNCTION
# ============================================================

def predict_probability(data):

    data_df = pd.DataFrame(
        data,
        columns=features
    )

    return model.predict_proba(data_df)


# ============================================================
# CREATE LIME EXPLAINER
# ============================================================

print("\nCreating LIME explainer...")

explainer = LimeTabularExplainer(
    X.values,
    feature_names=features,
    class_names=[
        "Low Risk",
        "High Risk"
    ],
    mode="classification",
    random_state=42
)


# ============================================================
# SELECT CUSTOMER
# ============================================================

customer_position = 0

customer_id = df.iloc[
    customer_position
]["customer_id"]

customer_data = X.iloc[
    customer_position
].values


print(
    f"\nExplaining customer: "
    f"{customer_id}"
)


# ============================================================
# CREATE EXPLANATION
# ============================================================

explanation = explainer.explain_instance(
    customer_data,
    predict_probability,
    num_features=10
)


# ============================================================
# PRINT EXPLANATION
# ============================================================

print("\n" + "=" * 60)
print("LIME EXPLANATION")
print("=" * 60)

print(
    f"\nCustomer ID: {customer_id}"
)

print("\nFeature contributions:")

for feature, weight in explanation.as_list():

    print(
        f"{feature}: {weight:.6f}"
    )


# ============================================================
# SAVE HTML EXPLANATION
# ============================================================

html_file = (
    OUTPUT_DIR
    / "lime_customer_explanation.html"
)

explanation.save_to_file(
    str(html_file)
)


# ============================================================
# SAVE CSV EXPLANATION
# ============================================================

lime_results = pd.DataFrame(
    explanation.as_list(),
    columns=[
        "Feature",
        "Contribution"
    ]
)

csv_file = (
    OUTPUT_DIR
    / "lime_customer_explanation.csv"
)

lime_results.to_csv(
    csv_file,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("LIME EXPLAINABILITY COMPLETED")
print("=" * 60)

print(
    f"\nHTML explanation saved to:\n"
    f"{html_file}"
)

print(
    f"\nCSV explanation saved to:\n"
    f"{csv_file}"
)