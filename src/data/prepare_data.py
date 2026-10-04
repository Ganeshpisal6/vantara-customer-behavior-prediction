from pathlib import Path

import pandas as pd

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

df_2009 = pd.read_excel(
    RAW_FILE,
    sheet_name="Year 2009-2010"
)

df_2010 = pd.read_excel(
    RAW_FILE,
    sheet_name="Year 2010-2011"
)

df = pd.concat(
    [df_2009, df_2010],
    ignore_index=True
)

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
            "nunique",
        ),
         last_purchase_date=(
         "invoicedate",
          "max",
        ),
        
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
