from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing_extensions import Annotated
from datetime import datetime

County = Annotated[int, Field(ge=0, le=15)]
Product_Type = Annotated[int, Field(ge=0, le=3)]

class PredictionData(BaseModel):
    county: County
    is_business: bool
    product_type: Product_Type
    is_consumption: bool
    timestamp: datetime


app = FastAPI()
items = []


@app.get("/")
def root():
    return "Energy Consumption and Production Predictions"


@app.post("/predict")
def predict(data: PredictionData):
    target = data.county * data.product_type
    return {"prediction": target}