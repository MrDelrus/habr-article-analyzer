from fastapi import APIRouter

from backend.config import settings
from backend.utils.s3_loader import list_models
from core.schemas.api.models import ModelItem, ModelListResponse

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelListResponse)
async def get_models() -> ModelListResponse:
    models = list_models()
    cleaned_models = [
        ModelItem(name=model.replace(f".{settings.MODEL_EXTENSION}", ""))
        for model in models
    ]
    return ModelListResponse(models=cleaned_models)
