import streamlit as st


st.set_page_config(
    page_title="Vantara",
    page_icon="📊",
    layout="wide"
)


st.title("Vantara")

st.subheader("Customer Churn & Customer Value Analytics")

st.write(
    """
    Vantara is a customer analytics platform that helps
    identify customers at risk of churn, understand
    customer value, and support retention decisions.
    """
)

st.divider()

st.info(
    "Use the pages in the sidebar to access the Vantara dashboard."
)