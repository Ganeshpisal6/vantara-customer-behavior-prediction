from sqlalchemy import Column, Float, Integer, String

from api.database import Base


class CustomerPrediction(Base):
    __tablename__ = "customer_predictions"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, index=True)
    churn_prediction = Column(Integer)
    churn_status = Column(String)
    churn_probability = Column(Float)