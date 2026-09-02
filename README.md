# Vantara - Customer Behavior Prediction

## Project Overview

Vantara is an AI-powered customer behavior prediction and analytics application designed to help businesses understand customer behavior, identify churn risk, analyze Customer Lifetime Value (CLV), and generate actionable business recommendations.

The project combines Machine Learning, Streamlit, FastAPI, and Docker to provide an interactive and deployable customer analytics solution.

## Key Features

- Customer behavior analysis
- Customer segmentation
- Churn risk prediction
- Churn probability analysis
- Customer Lifetime Value (CLV) analysis
- Business recommendations
- Action priority classification
- Interactive Streamlit dashboard
- Customer ID search
- Customer-level details
- FastAPI prediction API
- Batch CSV prediction support
- Docker and Docker Compose support

## Machine Learning

The project uses a Random Forest classification model for customer churn prediction.

The model uses customer behavioral features including:

- Recency
- Frequency
- Monetary value
- Total quantity
- Average order value
- Unique products
- RFM-related scores

The system classifies customers according to their churn risk and generates business recommendations based on customer value and risk.

## Customer Analytics

The dashboard provides:

- Total customers
- Total revenue
- Average CLV
- High-risk customer count
- Customer segment distribution
- Churn status distribution
- Churn risk leaderboard
- Customer search and filtering
- Customer recommendations

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit
- FastAPI
- Uvicorn
- Joblib
- SHAP
- Docker
- Docker Compose

## Project Structure

```text
customer-behavior-prediction/
│
├── api/
│   ├── main.py
│   ├── routers/
│   └── schemas/
│
├── config/
│
├── data/
│   └── processed/
│
├── docs/
│
├── frontend/
│   ├── app.py
│   └── pages/
│       ├── dashboard.py
│       ├── customer_details.py
│       └── prediction.py
│
├── models_artifacts/
│
├── notebooks/
│
├── src/
│   ├── data/
│   ├── features/
│   └── models/
│
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md