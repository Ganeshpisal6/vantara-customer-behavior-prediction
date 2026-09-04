from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

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
print("VANTARA - CUSTOMER SEGMENTATION")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Customers loaded: {len(df):,}")


# ============================================================
# FEATURES FOR CLUSTERING
# ============================================================

features = [
    "recency",
    "frequency",
    "monetary",
    "total_quantity",
    "average_order_value",
    "unique_products",
    "clv_score"
]

X = df[features].copy()


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

X = X.fillna(
    X.median()
)


# ============================================================
# SCALE FEATURES
# ============================================================

print("\nScaling features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# Save scaler

scaler_file = (
    MODEL_DIR
    / "clustering_scaler.pkl"
)

joblib.dump(
    scaler,
    scaler_file
)


# ============================================================
# K-MEANS
# ============================================================

print("\n" + "=" * 60)
print("K-MEANS CLUSTERING")
print("=" * 60)

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(
    X_scaled
)


kmeans_silhouette = silhouette_score(
    X_scaled,
    kmeans_labels
)

kmeans_db = davies_bouldin_score(
    X_scaled,
    kmeans_labels
)


print(f"Silhouette Score     : {kmeans_silhouette:.4f}")
print(f"Davies-Bouldin Score : {kmeans_db:.4f}")


# Save K-Means model

kmeans_file = (
    MODEL_DIR
    / "kmeans_model.pkl"
)

joblib.dump(
    kmeans,
    kmeans_file
)


# ============================================================
# DBSCAN
# ============================================================

print("\n" + "=" * 60)
print("DBSCAN CLUSTERING")
print("=" * 60)

best_dbscan = None
best_labels = None
best_silhouette = -1
best_db_score = None
best_eps = None
best_min_samples = None


# Try multiple DBSCAN configurations

for eps in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0]:

    for min_samples in [5, 10, 15]:

        dbscan_test = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )

        labels = dbscan_test.fit_predict(
            X_scaled
        )

        # Remove noise points for evaluation

        mask = labels != -1

        valid_labels = labels[mask]
        valid_data = X_scaled[mask]

        n_clusters = len(
            set(valid_labels)
        )

        # Need at least 2 clusters

        if n_clusters >= 2 and len(valid_data) > n_clusters:

            silhouette = silhouette_score(
                valid_data,
                valid_labels
            )

            db_score = davies_bouldin_score(
                valid_data,
                valid_labels
            )

            if silhouette > best_silhouette:

                best_silhouette = silhouette
                best_db_score = db_score
                best_dbscan = dbscan_test
                best_labels = labels
                best_eps = eps
                best_min_samples = min_samples


# Check whether a valid configuration was found

if best_dbscan is not None:

    dbscan = best_dbscan
    dbscan_labels = best_labels

    n_clusters = len(
        set(dbscan_labels) - {-1}
    )

    n_noise = list(
        dbscan_labels
    ).count(-1)

    print(f"Best EPS          : {best_eps}")
    print(f"Best Min Samples  : {best_min_samples}")
    print(f"Clusters found    : {n_clusters}")
    print(f"Noise points      : {n_noise}")

    print(
        f"Silhouette Score     : "
        f"{best_silhouette:.4f}"
    )

    print(
        f"Davies-Bouldin Score : "
        f"{best_db_score:.4f}"
    )

else:

    print(
        "No valid DBSCAN configuration "
        "produced at least 2 clusters."
    )

    dbscan = DBSCAN(
        eps=0.5,
        min_samples=5
    )

    dbscan_labels = dbscan.fit_predict(
        X_scaled
    )


# Save DBSCAN model

dbscan_file = (
    MODEL_DIR
    / "dbscan_model.pkl"
)

joblib.dump(
    dbscan,
    dbscan_file
)

# ============================================================
# ADD CLUSTERS TO DATASET
# ============================================================

df["KMeans_Cluster"] = kmeans_labels

df["DBSCAN_Cluster"] = dbscan_labels


# ============================================================
# BUSINESS-READABLE K-MEANS SEGMENTS
# ============================================================

cluster_summary = (
    df.groupby("KMeans_Cluster")
    .agg(
        Customers=("customer_id", "count"),
        Avg_Recency=("recency", "mean"),
        Avg_Frequency=("frequency", "mean"),
        Avg_Monetary=("monetary", "mean"),
        Avg_CLV=("clv_score", "mean")
    )
    .reset_index()
)


def assign_segment(row):

    if (
        row["Avg_Monetary"] >=
        cluster_summary["Avg_Monetary"].median()
        and
        row["Avg_CLV"] >=
        cluster_summary["Avg_CLV"].median()
    ):
        return "High Value Customers"

    elif row["Avg_Recency"] <= cluster_summary["Avg_Recency"].median():

        return "Active Customers"

    elif row["Avg_Recency"] > cluster_summary["Avg_Recency"].median():

        return "At Risk Customers"

    else:

        return "Regular Customers"


cluster_summary["Business_Segment"] = (
    cluster_summary.apply(
        assign_segment,
        axis=1
    )
)


print("\n" + "=" * 60)
print("BUSINESS SEGMENTS")
print("=" * 60)

print(
    cluster_summary.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

cluster_file = (
    BASE_DIR
    / "data"
    / "processed"
    / "customer_clusters.csv"
)

df.to_csv(
    cluster_file,
    index=False
)


summary_file = (
    BASE_DIR
    / "data"
    / "processed"
    / "cluster_summary.csv"
)

cluster_summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("CLUSTERING COMPLETED")
print("=" * 60)

print(
    f"\nCustomer clusters saved to:\n"
    f"{cluster_file}"
)

print(
    f"\nCluster summary saved to:\n"
    f"{summary_file}"
)

print(
    f"\nModels saved in:\n"
    f"{MODEL_DIR}"
)

print("\nSegmentation completed successfully.")