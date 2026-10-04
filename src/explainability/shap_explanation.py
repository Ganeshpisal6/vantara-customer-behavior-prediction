from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

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
    / "shap_outputs"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("VANTARA - SHAP EXPLAINABILITY")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

print(f"\nCustomers loaded: {len(df):,}")


# ============================================================
# CURRENT XGBOOST MODEL FEATURES
# ============================================================

features = [
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

X = df[features].copy()

for column in features:
    X[column] = pd.to_numeric(X[column], errors="coerce")

X = X.fillna(0)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading churn model...")

model = joblib.load(MODEL_FILE)

print(f"Model loaded: {type(model).__name__}")


# ============================================================
# SHAP EXPLAINER
# ============================================================

print("\nCreating SHAP explainer...")

explainer = shap.TreeExplainer(model)


# ============================================================
# SAMPLE FOR GLOBAL SHAP
# ============================================================

sample_size = min(500, len(X))

X_sample = X.sample(
    n=sample_size,
    random_state=42
)

shap_values = explainer.shap_values(X_sample)


# ============================================================
# HANDLE SHAP OUTPUT
# ============================================================

if isinstance(shap_values, list):
    shap_values_positive = shap_values[1]
else:
    shap_values_positive = shap_values


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

print("\nCalculating global feature importance...")

mean_abs_shap = abs(shap_values_positive).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": features,
    "Mean_Absolute_SHAP": mean_abs_shap
})

importance_df = importance_df.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=False
)

importance_file = (
    OUTPUT_DIR
    / "shap_feature_importance.csv"
)

importance_df.to_csv(
    importance_file,
    index=False
)


# ============================================================
# SHAP SUMMARY PLOT
# ============================================================

print("Creating SHAP summary plot...")

plt.figure()

shap.summary_plot(
    shap_values_positive,
    X_sample,
    feature_names=features,
    show=False
)

plt.tight_layout()

summary_file = (
    OUTPUT_DIR
    / "shap_summary.png"
)

plt.savefig(
    summary_file,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SHAP BAR PLOT
# ============================================================

print("Creating SHAP feature importance plot...")

plt.figure()

shap.summary_plot(
    shap_values_positive,
    X_sample,
    feature_names=features,
    plot_type="bar",
    show=False
)

plt.tight_layout()

bar_file = (
    OUTPUT_DIR
    / "shap_feature_importance.png"
)

plt.savefig(
    bar_file,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# CUSTOMER-SPECIFIC SHAP FUNCTION
# ============================================================
def create_customer_shap(customer_id):
    """
    Create a SHAP explanation for a specific customer ID.
    """

    customer_id = str(customer_id).strip()

    customer_mask = (
        df["customer_id"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        == customer_id
    )

    customer_rows = df.loc[customer_mask]

    if customer_rows.empty:
        print(f"Customer {customer_id} not found.")
        return None

    customer_row = customer_rows.iloc[0]

    X_customer = customer_row[features].to_frame().T

    for column in features:
        X_customer[column] = pd.to_numeric(
            X_customer[column],
            errors="coerce"
        )

    X_customer = X_customer.fillna(0).astype(float)
    customer_shap = explainer.shap_values(X_customer)

    if isinstance(customer_shap, list):
        customer_shap = customer_shap[1]

    customer_shap = np.asarray(customer_shap).reshape(-1)

    expected_value = explainer.expected_value

    if isinstance(expected_value, (list, np.ndarray)):
        base_value = float(np.asarray(expected_value).reshape(-1)[0])
    else:
        base_value = float(expected_value)

    explanation = shap.Explanation(
        values=customer_shap,
        base_values=base_value,
        data=np.asarray(X_customer.iloc[0]).reshape(-1),
        feature_names=features
    )

    output_file = (
        OUTPUT_DIR
        / f"customer_{customer_id}_shap.png"
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(
        explanation,
        max_display=len(features),
        show=False
    )
    plt.title(f"SHAP Explanation — Customer {customer_id}")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved customer SHAP explanation: {output_file}")

    return output_file

# ============================================================
# REQUIRED REPRESENTATIVE CUSTOMERS
# ============================================================

print("\nCreating representative customer explanations...")

available_customers = (
    df["customer_id"]
    .astype(str)
    .str.replace(".0", "", regex=False)
)

representative_ids = []

for customer_id in ["12346", "12347", "12348"]:

    if customer_id in available_customers.values:

        representative_ids.append(customer_id)


# If those IDs are unavailable, use first available customers
if len(representative_ids) < 3:

    for value in available_customers:

        if value not in representative_ids:

            representative_ids.append(value)

        if len(representative_ids) == 3:

            break


for customer_id in representative_ids:

    create_customer_shap(customer_id)


# ============================================================
# SAVE SAMPLE SHAP VALUES
# ============================================================

shap_values_df = pd.DataFrame(
    shap_values_positive,
    columns=features
)

shap_values_file = (
    OUTPUT_DIR
    / "shap_values.csv"
)

shap_values_df.to_csv(
    shap_values_file,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("SHAP EXPLAINABILITY COMPLETED")
print("=" * 60)

print(
    f"\nSHAP outputs saved to:\n{OUTPUT_DIR}"
)

print("\nCreated:")
print("- shap_feature_importance.csv")
print("- shap_summary.png")
print("- shap_feature_importance.png")
print("- shap_values.csv")

for customer_id in representative_ids:

    print(
        f"- customer_{customer_id}_shap.png"
    )
