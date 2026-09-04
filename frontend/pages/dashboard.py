from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Recommendations CSV is inside data/raw
DATA_FILE = BASE_DIR / "data" / "raw" / "vantara_recommendations.csv"

@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        st.error(f"Data file not found: {DATA_FILE}")
        st.stop()

    return pd.read_csv(DATA_FILE)

df = load_data()
df["Customer ID"] = pd.to_numeric(
    df["Customer ID"],
    errors="coerce"
).astype("Int64").astype(str)
# ============================================================
# TITLE
# ============================================================

st.title("Vantara Customer Intelligence Platform")

st.write(
    "AI-powered customer analytics for churn prediction, "
    "customer segmentation, CLV analysis and business recommendations."
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Filters")

# Customer Segment filter

segments = ["All"] + sorted(
    df["Customer_Segment"]
    .dropna()
    .unique()
    .tolist()
)

selected_segment = st.sidebar.selectbox(
    "Customer Segment",
    segments
)

# Churn filter

churn_options = ["All"] + sorted(
    df["Churn_Status"]
    .dropna()
    .unique()
    .tolist()
)

selected_churn = st.sidebar.selectbox(
    "Churn Status",
    churn_options
)

# Country filter

countries = ["All"] + sorted(
    df["Country"]
    .dropna()
    .unique()
    .tolist()
)

selected_country = st.sidebar.selectbox(
    "Country",
    countries
)


# CLV Category filter

clv_categories = ["All"] + sorted(
    df["CLV_Category"]
    .dropna()
    .unique()
    .tolist()
)

selected_clv = st.sidebar.selectbox(
    "Value Tier",
    clv_categories
)



# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df.copy()

if selected_segment != "All":

    filtered_df = filtered_df[
        filtered_df["Customer_Segment"]
        == selected_segment
    ]

if selected_churn != "All":

    filtered_df = filtered_df[
        filtered_df["Churn_Status"]
        == selected_churn
    ]

if selected_country != "All":

    filtered_df = filtered_df[
        filtered_df["Country"] == selected_country
    ]


if selected_clv != "All":

    filtered_df = filtered_df[
        filtered_df["CLV_Category"] == selected_clv
    ]

# ============================================================
# CUSTOMER SEARCH  13.1  
# ============================================================

st.sidebar.subheader("Customer Search")

customer_id_search = st.sidebar.text_input(
    "Enter Customer ID",
    placeholder="Example: 12346"
)

if customer_id_search:
    customer_id = customer_id_search.strip()

    filtered_df = filtered_df[
        filtered_df["Customer ID"] == customer_id
    ]

    

# ============================================================
# CUSTOMER DETAILS 13.2
# ============================================================

st.subheader("Customer Details")

if customer_id_search:
    if not filtered_df.empty:

        customer = filtered_df.iloc[0]

        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            st.metric(
                "Customer ID",
                customer["Customer ID"]
            )

        with col2:
            st.metric(
                "Customer Segment",
                customer["Customer_Segment"]
            )

        with col3:
            st.metric(
                "Churn Status",
                customer["Churn_Status"]
            )

        st.write("### Customer Information")

        st.write(
            f"**Recency:** {customer['Recency']}"
        )

        st.write(
            f"**Frequency:** {customer['Frequency']}"
        )

        st.write(
            f"**Monetary:** {customer['Monetary']}"
        )

        st.write(
            f"**CLV Category:** {customer['CLV_Category']}"
        )

        st.write(
            f"**Business Recommendation:** "
            f"{customer['Business_Recommendation']}"
        )

        st.write(
            f"**Action Priority:** "
            f"{customer['Action_Priority']}"
        )

    else:
        st.warning("Customer ID not found.")



# ============================================================
# KPI SECTION
# ============================================================

col1, col2, col3, col4 = st.columns([1, 1.5, 1, 1])

with col1:

    st.metric(
        "Total Customers",
        len(filtered_df)
    )

with col2:

    st.metric(
        "Total Revenue",
        f"£{filtered_df['Monetary'].sum():,.0f}"
    )

with col3:

    st.metric(
        "Average CLV",
        f"£{filtered_df['CLV_Score'].mean():,.2f}"
    )

with col4:

    high_risk = (
        filtered_df["Churn_Status"]
        == "High Risk"
    ).sum()

    st.metric(
        "High Risk Customers",
        high_risk
    )

    
# CHARTS
# ============================================================

col1, col2 = st.columns(2)

# LEFT: BAR CHART
with col1:
    st.subheader("Customer Segments")

    segment_counts = (
        filtered_df["Customer_Segment"]
        .value_counts()
        .reset_index()
    )

    segment_counts.columns = [
        "Customer_Segment",
        "Customers"
    ]

    fig = px.bar(
        segment_counts,
        x="Customer_Segment",
        y="Customers",
        title="Customers by Segment"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# RIGHT: PIE CHART
with col2:
    st.subheader("Churn Status")

    churn_counts = (
        filtered_df["Churn_Status"]
        .value_counts()
        .reset_index()
    )

    churn_counts.columns = [
        "Churn_Status",
        "Customers"
    ]

    fig = px.pie(
        churn_counts,
        names="Churn_Status",
        values="Customers",
        title="Churn Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ============================================================
# CHURN RISK LEADERBOARD
# ============================================================

st.subheader("Churn Risk Leaderboard")

leaderboard_columns = [
    "Customer ID",
    "Churn_Probability",
    "Churn_Status",
    "CLV_Score",
    "Action_Priority"
]

leaderboard_df = (
    filtered_df[leaderboard_columns]
    .sort_values(
        by="Churn_Probability",
        ascending=False
    )
    .head(10)
)

st.dataframe(
    leaderboard_df,
    use_container_width=True
)

# ============================================================
# CLV ANALYSIS
# ============================================================

st.subheader("Customer Lifetime Value")

clv_counts = (
    filtered_df["CLV_Category"]
    .value_counts()
    .reset_index()
)

clv_counts.columns = [
    "CLV_Category",
    "Customers"
]

fig = px.bar(
    clv_counts,
    x="CLV_Category",
    y="Customers",
    title="CLV Category Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ============================================================
# BUSINESS RECOMMENDATIONS
# ============================================================

st.subheader("Business Recommendations")

recommendation_counts = (
    filtered_df["Business_Recommendation"]
    .value_counts()
    .reset_index()
)

recommendation_counts.columns = [
    "Recommendation",
    "Customers"
]

st.dataframe(
    recommendation_counts,
    use_container_width=True
)

# ============================================================
# DOWNLOAD CUSTOMER DATA
# ============================================================

csv_data = filtered_df.to_csv(index=False)

st.download_button(
    label="Download Customer Data",
    data=csv_data,
    file_name="vantara_customer_data.csv",
    mime="text/csv"
)

# ============================================================
# CUSTOMER DATA
# ============================================================

st.subheader("Customer Intelligence Data")

display_columns = [
    "Customer ID",
    "Customer_Segment",
    "Churn_Status",
    "Churn_Probability",
    "CLV_Score",
    "CLV_Category",
    "Action_Priority",
    "Business_Recommendation"
]

st.dataframe(
    filtered_df[display_columns],
    use_container_width=True
)

# ============================================================
# STEP 14 - BUSINESS INSIGHTS
# ============================================================

st.header("Business Insights & Recommendations")

# ------------------------------------------------------------
# CUSTOMER INSIGHTS
# ------------------------------------------------------------

total_customers = len(filtered_df)

high_risk_customers = 0

if "Churn_Status" in filtered_df.columns:
    high_risk_customers = (
        filtered_df["Churn_Status"] == "High Risk"
    ).sum()

total_revenue = 0

if "Monetary" in filtered_df.columns:
    total_revenue = filtered_df["Monetary"].sum()

average_clv = 0

if "CLV_Score" in filtered_df.columns:
    average_clv = filtered_df["CLV_Score"].mean()


# ============================================================
# INSIGHT 1 - CHURN
# ============================================================

st.subheader("Customer Retention Insight")

if high_risk_customers > 0:

    st.warning(
        f"{high_risk_customers} customers are currently classified "
        f"as High Risk. These customers should receive retention "
        f"campaigns, personalized offers and follow-up communication."
    )

else:

    st.success(
        "No high-risk customers were found in the current selection."
    )


# ============================================================
# INSIGHT 2 - REVENUE
# ============================================================

st.subheader("Revenue Insight")

st.info(
    f"The selected customers generated approximately "
    f"£{total_revenue:,.2f} in total revenue."
)


# ============================================================
# INSIGHT 3 - CUSTOMER VALUE
# ============================================================

st.subheader("Customer Value Insight")

st.info(
    f"The average Customer Lifetime Value (CLV) "
    f"for the selected customers is "
    f"{average_clv:,.2f}."
)


# ============================================================
# INSIGHT 4 - BUSINESS ACTION
# ============================================================

st.subheader("Recommended Business Actions")

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("### 🔴 High Risk")

    st.write(
        "• Send retention offers\n\n"
        "• Provide personalized discounts\n\n"
        "• Contact inactive customers\n\n"
        "• Identify reasons for churn"
    )


with col2:

    st.markdown("### 🟠 Medium Priority")

    st.write(
        "• Increase customer engagement\n\n"
        "• Recommend relevant products\n\n"
        "• Send personalized campaigns\n\n"
        "• Encourage repeat purchases"
    )


with col3:

    st.markdown("### 🟢 Low Risk")

    st.write(
        "• Maintain customer relationship\n\n"
        "• Offer loyalty rewards\n\n"
        "• Cross-sell products\n\n"
        "• Encourage referrals"
    )


# ============================================================
# FINAL PROJECT MESSAGE
# ============================================================

st.success(
    "Vantara Customer Intelligence Platform is successfully "
    "using customer behavior, churn prediction, CLV analysis "
    "and business recommendations to support data-driven "
    "decision making."
)