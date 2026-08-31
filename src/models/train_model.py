import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "customer_model_features.csv"
)

MODEL_DIR = BASE_DIR / "models_artifacts"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("Loading model dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Customers loaded: {len(df):,}")


# ------------------------------------------------------------
# CREATE TARGET
# ------------------------------------------------------------

df["churn_target"] = (
    df["churn_status"] == "High Risk"
).astype(int)


# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------

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

X = df[features]

y = df["churn_target"]


# ------------------------------------------------------------
# TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print(f"Training rows: {len(X_train):,}")
print(f"Testing rows: {len(X_test):,}")


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# PREDICTIONS
# ------------------------------------------------------------

y_pred = model.predict(X_test)


# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    f"\nModel Accuracy: {accuracy:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ------------------------------------------------------------
# SAVE MODEL
# ------------------------------------------------------------

model_file = (
    MODEL_DIR
    / "churn_model.pkl"
)

joblib.dump(
    model,
    model_file
)


print(
    f"\nModel saved to:"
    f"\n{model_file}"
)