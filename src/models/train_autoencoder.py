from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "processed" / "customer_model_features.csv"
MODEL_DIR = BASE_DIR / "models_artifacts"

MODEL_FILE = MODEL_DIR / "autoencoder.h5"
SCALER_FILE = MODEL_DIR / "autoencoder_scaler.pkl"

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


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    X = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    input_dim = X_scaled.shape[1]

    inputs = Input(shape=(input_dim,))

    encoded = Dense(32, activation="relu")(inputs)
    encoded = Dense(16, activation="relu")(encoded)
    encoded = Dense(8, activation="relu")(encoded)

    decoded = Dense(16, activation="relu")(encoded)
    decoded = Dense(32, activation="relu")(decoded)
    outputs = Dense(input_dim, activation="linear")(decoded)

    autoencoder = Model(inputs, outputs)

    autoencoder.compile(
        optimizer="adam",
        loss="mse",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    print("Training Autoencoder...")

    autoencoder.fit(
        X_scaled,
        X_scaled,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1,
    )

    autoencoder.save(MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    print("\nAutoencoder model saved successfully.")
    print(f"Model: {MODEL_FILE}")
    print(f"Scaler: {SCALER_FILE}")


if __name__ == "__main__":
    main()