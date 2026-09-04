from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "processed" / "customer_model_features.csv"
OUTPUT_FILE = BASE_DIR / "models_artifacts" / "cross_validation_results.csv"

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
    df = pd.read_csv(INPUT_FILE)

    X = df[FEATURES].fillna(0)

    y = (df["churn_status"] == "High Risk").astype(int)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=500, random_state=42)),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]),
    }

    results = []

    for name, model in models.items():
        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="roc_auc",
        )

        results.append({
            "Model": name,
            "Fold_1_ROC_AUC": scores[0],
            "Fold_2_ROC_AUC": scores[1],
            "Fold_3_ROC_AUC": scores[2],
            "Fold_4_ROC_AUC": scores[3],
            "Fold_5_ROC_AUC": scores[4],
            "Mean_ROC_AUC": scores.mean(),
            "Std_ROC_AUC": scores.std(),
        })

        print(f"\n{name}")
        print(f"Fold ROC-AUC: {scores}")
        print(f"Mean ROC-AUC: {scores.mean():.4f}")
        print(f"Std ROC-AUC : {scores.std():.4f}")

    results_df = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print("\n5-fold cross-validation completed.")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()