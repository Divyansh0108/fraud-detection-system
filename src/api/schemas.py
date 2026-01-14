from pydantic import BaseModel, Field
from typing import List

class Transaction(BaseModel):
    Time: float = Field(..., description="Time in seconds since first transaction")
    Amount: float = Field(..., description="Transaction amount")
    V_features: List[float] = Field(
        ..., min_length=28, max_length=28, description="PCA components V1-V28"
    )

class PredictionResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    probability: float
    latency_ms: float