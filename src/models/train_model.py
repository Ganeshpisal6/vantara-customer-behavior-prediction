from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


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
# LOAD DATA
# ============================================================

print("=" * 60)
print("VANTARA - CLASSICAL ML MODEL COMPARISON")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Customers loaded: {len(df):,}")


# ============================================================
# CREATE TARGET
# ============================================================

df["churn_target"] = (
    df["churn_status"] == "High Risk"
).astype(int)


print("\nTarget distribution:")
print(df["churn_target"].value_counts())


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

X = df[features]

y = df["churn_target"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining rows: {len(X_train):,}")
print(f"Testing rows: {len(X_test):,}")


# ============================================================
# DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        verbosity=-1
    ),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            KNeighborsClassifier(
                n_neighbors=5
            )
        )
    ])
}


# ============================================================
# TRAIN MODELS
# ============================================================

results = []

best_model = None
best_model_name = None
best_auc = 0


for name, model in models.items():

    print("\n" + "=" * 60)
    print(f"Training: {name}")
    print("=" * 60)

    try:

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):

            y_probability = model.predict_proba(X_test)[:, 1]

        else:

            y_probability = None


        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        if y_probability is not None:

            auc = roc_auc_score(
                y_test,
                y_probability
            )

        else:

            auc = 0


        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC-AUC  : {auc:.4f}")


        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                y_pred,
                zero_division=0
            )
        )


        results.append({

            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC-AUC": auc

        })


        # Save individual model

        model_filename = (
            name.lower()
            .replace(" ", "_")
            .replace("-", "_")
            + ".pkl"
        )

        model_path = MODEL_DIR / model_filename

        joblib.dump(
            model,
            model_path
        )


        # Select best model based on ROC-AUC

        if auc > best_auc:

            best_auc = auc

            best_model = model

            best_model_name = name


    except Exception as e:

        print(f"\nERROR while training {name}:")
        print(e)


# ============================================================
# MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="ROC-AUC",
    ascending=False
)


comparison_file = (
    MODEL_DIR
    / "model_comparison.csv"
)

results_df.to_csv(
    comparison_file,
    index=False
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

if best_model is not None:

    best_model_file = (
        MODEL_DIR
        / "churn_model.pkl"
    )

    joblib.dump(
        best_model,
        best_model_file
    )

    print("\n" + "=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print(f"Model   : {best_model_name}")
    print(f"ROC-AUC : {best_auc:.4f}")

    print(
        f"\nBest model saved to:\n"
        f"{best_model_file}"
    )


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)

print(
    f"\nComparison saved to:\n"
    f"{comparison_file}"
)

print("\nTraining completed successfully.")