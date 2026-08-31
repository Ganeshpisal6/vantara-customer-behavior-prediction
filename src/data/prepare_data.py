import pandas as pd
from pathlib import Path


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_FILE = BASE_DIR / "data" / "raw" / "online_retail_II.xlsx"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("Loading raw data...")

df = pd.read_excel(RAW_FILE)

print(f"Original rows: {len(df):,}")


# ------------------------------------------------------------
# CLEAN COLUMN NAMES
# ------------------------------------------------------------

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)


# ------------------------------------------------------------
# REMOVE INVALID RECORDS
# ------------------------------------------------------------

df = df.dropna(subset=["customer_id"])

df = df[df["quantity"] > 0]

df = df[df["price"] > 0]

df["invoicedate"] = pd.to_datetime(
    df["invoicedate"],
    errors="coerce"
)

df = df.dropna(subset=["invoicedate"])


# ------------------------------------------------------------
# CREATE REVENUE
# ------------------------------------------------------------

df["revenue"] = df["quantity"] * df["price"]


# ------------------------------------------------------------
# CUSTOMER-LEVEL FEATURES
# ------------------------------------------------------------

reference_date = df["invoicedate"].max() + pd.Timedelta(days=1)

customer_features = (
    df.groupby("customer_id")
    .agg(
        recency=(
            "invoicedate",
            lambda x: (reference_date - x.max()).days
        ),
        frequency=(
            "invoice",
            "nunique"
        ),
        monetary=(
            "revenue",
            "sum"
        ),
        total_quantity=(
            "quantity",
            "sum"
        ),
        average_order_value=(
            "revenue",
            "mean"
        ),
        unique_products=(
            "stockcode",
            "nunique"
        )
    )
    .reset_index()
)


# ------------------------------------------------------------
# SAVE CUSTOMER FEATURES
# ------------------------------------------------------------

output_file = PROCESSED_DIR / "customer_features.csv"

customer_features.to_csv(
    output_file,
    index=False
)


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\nData preparation completed!")

print(
    f"Customers created: "
    f"{len(customer_features):,}"
)

print(
    f"Saved to: "
    f"{output_file}"
)

print("\nColumns:")
print(customer_features.columns.tolist())

print("\nFirst 5 rows:")
print(customer_features.head())
