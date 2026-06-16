from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing_extensions import Annotated
from datetime import datetime
import pandas as pd
from contextlib import asynccontextmanager

from app.core.model import model_instance

County = Annotated[int, Field(ge=0, le=15, description="County ID (0-15)")]
ProductType = Annotated[
    int, Field(ge=0, le=3, description="Product Type ID (0-3)")
]


# Common validation to strip timezone to match the naive format in history
def strip_timezone(v: datetime) -> datetime:
    return v.replace(tzinfo=None) if v.tzinfo else v


class PredictionData(BaseModel):
    county: County
    is_business: bool
    product_type: ProductType
    is_consumption: bool
    timestamp: datetime = Field(
        ...,
        examples=["2023-04-01T12:00:00"],
        description="Naive datetime (no timezone), hourly resolution",
    )
    eic_count: int = Field(
        ..., description="Number of consumption points", gt=0
    )
    installed_capacity: float = Field(
        ..., description="Installed solar capacity (kW)", ge=0
    )

    @field_validator("timestamp")
    @classmethod
    def apply_strip_timezone(cls, v: datetime) -> datetime:
        return strip_timezone(v)


class RangePredictionRequest(BaseModel):
    county: County
    is_business: bool
    product_type: ProductType
    is_consumption: bool
    start_datetime: datetime = Field(
        ...,
        examples=["2023-04-01T00:00:00"],
        description="Start naive datetime (no timezone)",
    )
    end_datetime: datetime = Field(
        ...,
        examples=["2023-04-05T00:00:00"],
        description="End naive datetime (no timezone)",
    )

    @field_validator("start_datetime")
    @classmethod
    def strip_tz_start(cls, v: datetime) -> datetime:
        return strip_timezone(v)

    @field_validator("end_datetime")
    @classmethod
    def validate_range(cls, v: datetime, info):
        v = strip_timezone(v)
        if "start_datetime" in info.data:
            start_dt = info.data["start_datetime"]
            if v <= start_dt:
                raise ValueError("end_datetime must be after start_datetime")
            if (v - start_dt).days > 30:
                raise ValueError("Range cannot exceed 30 days")
        return v


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_instance.load()
    yield


app = FastAPI(title="Enefit Energy Forecasting API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit default port
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Energy Consumption and Production Predictions API"}


@app.get("/health")
def health():
    if not model_instance.initialized:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict(data: PredictionData):
    input_data = data.model_dump()
    # Pydantic v2 keeps datetime objects; model.py expects basic
    # strings/timestamps or pandas handles it
    input_data["timestamp"] = input_data["timestamp"].isoformat()
    input_data["is_business"] = int(input_data["is_business"])
    input_data["is_consumption"] = int(input_data["is_consumption"])

    try:
        prediction = model_instance.predict(input_data)
        return {"prediction": prediction}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="Internal prediction error"
        )


@app.post("/predict/range")
def predict_range(data: RangePredictionRequest):
    try:
        params = {
            "county": data.county,
            "is_business": int(data.is_business),
            "product_type": data.product_type,
            "is_consumption": int(data.is_consumption),
        }

        # Generate hourly timestamps for the range
        timestamps = pd.date_range(
            start=data.start_datetime, end=data.end_datetime, freq="h"
        ).tolist()

        results = model_instance.predict_range(params, timestamps)
        return {"predictions": results, "count": len(results)}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
