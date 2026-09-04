from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import partial_dependence

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
    / "pdp_outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("VANTARA - PARTIAL DEPENDENCE ANALYSIS")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

print(f"\nCustomers loaded: {len(df):,}")


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

print(
    f"Model loaded: {type(model).__name__}"
)


# ============================================================
# IMPORTANT FEATURES
# ============================================================

important_features = [
    "recency",
    "frequency",
    "clv_score"
]


# ============================================================
# CREATE PDPs
# ============================================================


for feature in important_features:

    print(f"Creating PDP for: {feature}")

    feature_index = features.index(feature)

    result = partial_dependence(
        model,
        X,
        [feature_index],
        grid_resolution=20
    )

    # Your scikit-learn version returns:
    # (average_predictions, grid_values)

    average = result[0][0]
    grid = result[1][0]

    plt.figure(figsize=(8, 6))

    plt.plot(grid, average)

    plt.xlabel(
        feature.replace("_", " ").title()
    )

    plt.ylabel("Partial Dependence")

    plt.title(
        f"Partial Dependence Plot - {feature}"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    output_file = (
        OUTPUT_DIR / f"pdp_{feature}.png"
    )

    plt.savefig(
        output_file,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_file}")



# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("PDP ANALYSIS COMPLETED")
print("=" * 60)

print(
    f"\nPDP outputs saved to:\n"
    f"{OUTPUT_DIR}"
)

print("\nCreated:")

for feature in important_features:

    print(
        f"- pdp_{feature}.png"
    )