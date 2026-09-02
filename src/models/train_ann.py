from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "customer_model_features.csv"
)

MODEL_DIR = BASE_DIR / "models_artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


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


df = pd.read_csv(INPUT_FILE)

df["churn_target"] = (
    df["churn_status"] == "High Risk"
).astype(int)

X = df[FEATURES]
y = df["churn_target"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=300,
    random_state=42,
)


print("Training ANN...")

model.fit(
    X_train_scaled,
    y_train,
)


predictions = model.predict(
    X_test_scaled
)

probabilities = model.predict_proba(
    X_test_scaled
)[:, 1]


accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)


print("\nANN RESULTS")
print("=" * 50)
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


joblib.dump(
    model,
    MODEL_DIR / "ann_churn_model.pkl"
)

joblib.dump(
    scaler,
    MODEL_DIR / "ann_scaler.pkl"
)


print("\nANN model saved successfully.")