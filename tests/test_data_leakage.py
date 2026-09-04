from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "processed" / "customer_model_features.csv"

TARGET = "churn_status"

FEATURES = [
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
    "clv_score",
]

def test_no_target_column_in_features():
    assert TARGET not in FEATURES, (
        f"Target column '{TARGET}' is present in the feature list."
    )


def test_customer_id_is_not_a_feature():
    assert "customer_id" not in FEATURES


def test_feature_columns_exist():
    df = pd.read_csv(INPUT_FILE)

    missing = [feature for feature in FEATURES if feature not in df.columns]

    assert not missing, f"Missing feature columns: {missing}"


def test_no_duplicate_customer_ids():
    df = pd.read_csv(INPUT_FILE)

    assert not df["customer_id"].duplicated().any(), (
        "Duplicate customer IDs detected."
    )


if __name__ == "__main__":
    print("Running leakage tests...")

    test_no_target_column_in_features()
    test_customer_id_is_not_a_feature()
    test_feature_columns_exist()
    test_no_duplicate_customer_ids()

    print("All leakage tests passed.")