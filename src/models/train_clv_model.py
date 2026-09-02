from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# PATHS
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


# LOAD DATA
print("=" * 60)
print("VANTARA - CLV REGRESSION")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"Customers loaded: {len(df):,}")


# FEATURES
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
    "rfm_score"
]

X = df[features]

y = df["clv_score"]


# TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print(f"Training rows: {len(X_train):,}")
print(f"Testing rows: {len(X_test):,}")


# MODEL
print("\nTraining CLV Random Forest Regressor...")

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)


# PREDICTION
y_pred = model.predict(X_test)


# EVALUATION
mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


print("\n" + "=" * 60)
print("CLV REGRESSION RESULTS")
print("=" * 60)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# SAVE MODEL
model_file = (
    MODEL_DIR
    / "clv_model.pkl"
)

joblib.dump(
    model,
    model_file
)

print(
    f"\nCLV model saved to:\n{model_file}"
)

print("\nCLV regression completed successfully.")