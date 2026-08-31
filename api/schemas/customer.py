from pydantic import BaseModel


class CustomerFeatures(BaseModel):

    recency: float
    frequency: float
    monetary: float
    total_quantity: float
    average_order_value: float
    unique_products: float
    recency_score: float
    frequency_score: float
    monetary_score: float
    rfm_score: float
    clv_score: float