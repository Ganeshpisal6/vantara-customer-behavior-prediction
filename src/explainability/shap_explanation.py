from pathlib import Path

import joblib
import matplotlib.pyplot as plt
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

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("VANTARA - SHAP EXPLAINABILITY")
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
# SAMPLE DATA
# ============================================================

# Use a sample for faster SHAP calculation

sample_size = min(
    500,
    len(X)
)

X_sample = X.sample(
    n=sample_size,
    random_state=42
)


# ============================================================
# HANDLE PIPELINE MODELS
# ============================================================

model_for_shap = model
X_for_shap = X_sample

if hasattr(model, "named_steps"):

    print("\nPipeline detected.")

    model_for_shap = model.named_steps[
        "model"
    ]

    scaler = model.named_steps.get(
        "scaler"
    )

    if scaler is not None:

        X_for_shap = scaler.transform(
            X_sample
        )


# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

print("\nCreating SHAP explainer...")

explainer = shap.TreeExplainer(
    model_for_shap
)

shap_values = explainer.shap_values(
    X_for_shap
)


# ============================================================
# HANDLE BINARY CLASSIFICATION OUTPUT
# ============================================================

if isinstance(shap_values, list):

    shap_values_positive = shap_values[1]

else:

    shap_values_positive = shap_values


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

print("\nCalculating global feature importance...")

mean_abs_shap = abs(
    shap_values_positive
).mean(axis=0)


importance_df = pd.DataFrame({
    "Feature": features,
    "Mean_Absolute_SHAP": mean_abs_shap
})

importance_df = importance_df.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=False
)


print("\nSHAP Feature Importance:")

print(
    importance_df.to_string(
        index=False
    )
)


# Save feature importance

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

print("\nCreating SHAP summary plot...")

plt.figure()

shap.summary_plot(
    shap_values_positive,
    X_for_shap,
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

print("Creating SHAP bar plot...")

plt.figure()

shap.summary_plot(
    shap_values_positive,
    X_for_shap,
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
# INDIVIDUAL CUSTOMER EXPLANATIONS
# ============================================================

print("\nCreating individual customer explanations...")


# Select first 3 representative customers
representative_rows = X_sample.iloc[:3]


for position in range(3):

    # Get original row position
    sample_index = representative_rows.index[position]

    # Get customer ID
    customer_id = df.loc[
        sample_index,
        "customer_id"
    ]

    print(
        f"Creating explanation {position + 1} "
        f"for customer {customer_id}"
    )


    # Get feature row
    row = representative_rows.iloc[
        position
    ]


    # Get transformed row for SHAP
    if hasattr(X_for_shap, "iloc"):

        row_transformed = X_for_shap.iloc[
            position
        ].values

    else:

        row_transformed = X_for_shap[
            position
        ]


    # Get SHAP values
    row_shap = shap_values_positive[
        position
    ]


    # Get expected value
    expected_value = explainer.expected_value

    if hasattr(expected_value, "__len__"):

        base_value = expected_value[1]

    else:

        base_value = expected_value


    # Create SHAP explanation
    explanation = shap.Explanation(
        values=row_shap,
        base_values=base_value,
        data=row_transformed,
        feature_names=features
    )


    # Create waterfall plot
    plt.figure()

    shap.plots.waterfall(
        explanation,
        show=False
    )

    plt.tight_layout()


    customer_file = (
        OUTPUT_DIR
        / f"customer_explanation_{position + 1}.png"
    )


    plt.savefig(
        customer_file,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# SAVE SHAP VALUES
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
    f"\nSHAP outputs saved to:\n"
    f"{OUTPUT_DIR}"
)

print(
    "\nCreated:"
    "\n- shap_feature_importance.csv"
    "\n- shap_summary.png"
    "\n- shap_feature_importance.png"
    "\n- customer_explanation_1.png"
    "\n- customer_explanation_2.png"
    "\n- customer_explanation_3.png"
    "\n- shap_values.csv"
)

# ============================================================
# SAVE SHAP VALUES
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
    f"\nSHAP outputs saved to:\n"
    f"{OUTPUT_DIR}"
)

print(
    "\nCreated:"
    "\n- shap_feature_importance.csv"
    "\n- shap_summary.png"
    "\n- shap_feature_importance.png"
    "\n- customer_explanation_1.png"
    "\n- customer_explanation_2.png"
    "\n- customer_explanation_3.png"
    "\n- shap_values.csv"
)