from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# PATHS
# ============================================================

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


# ============================================================
# FEATURES
# ============================================================

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


# ============================================================
# LOAD DATA
# ============================================================

print("Loading customer data...")

df = pd.read_csv(INPUT_FILE)

df["churn_target"] = (
    df["churn_status"] == "High Risk"
).astype(int)

X = df[FEATURES]
y = df["churn_target"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# ============================================================
# SCALE DATA
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# RESHAPE FOR LSTM
# ============================================================

# LSTM expects:
# samples, timesteps, features

X_train_lstm = X_train_scaled.reshape(
    X_train_scaled.shape[0],
    1,
    X_train_scaled.shape[1]
)

X_test_lstm = X_test_scaled.reshape(
    X_test_scaled.shape[0],
    1,
    X_test_scaled.shape[1]
)


# ============================================================
# BUILD LSTM
# ============================================================

print("\nBuilding LSTM model...")

model = Sequential(
    [
        LSTM(
            32,
            input_shape=(
                1,
                len(FEATURES)
            )
        ),
        Dropout(0.2),
        Dense(
            16,
            activation="relu"
        ),
        Dense(
            1,
            activation="sigmoid"
        ),
    ]
)


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining LSTM...")

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)


model.fit(
    X_train_lstm,
    y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1,
)


# ============================================================
# PREDICTIONS
# ============================================================

probabilities = model.predict(
    X_test_lstm,
    verbose=0
).ravel()

predictions = (
    probabilities >= 0.5
).astype(int)


# ============================================================
# EVALUATION
# ============================================================

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


print("\n" + "=" * 50)
print("LSTM RESULTS")
print("=" * 50)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    MODEL_DIR / "lstm_churn_model.h5"
)

joblib.dump(
    scaler,
    MODEL_DIR / "lstm_scaler.pkl"
)


print("\nLSTM model saved successfully.")