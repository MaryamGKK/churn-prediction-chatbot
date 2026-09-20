from fastapi import APIRouter, Depends

from api.dependencies import get_predictor
from api.security import verify_api_key
from src.schemas import CustomerFeatures, PredictionResult

router = APIRouter()


@router.post("/predict", response_model=PredictionResult, dependencies=[Depends(verify_api_key)])
async def predict(features: CustomerFeatures):
    return get_predictor().predict(features)
