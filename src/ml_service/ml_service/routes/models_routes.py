from fastapi import APIRouter, Depends

from core.schemas.api import ModelItem, ModelListResponse
from ml_service import settings
from ml_service.client import list_models
from ml_service.security import verify_internal_key

router = APIRouter(
    prefix="/v0/models",
    tags=["models"],
    dependencies=[Depends(verify_internal_key)],
)


@router.get("", response_model=ModelListResponse)
async def get_models() -> ModelListResponse:

    models = list_models()
    cleaned_models = [
        ModelItem(name=model.replace(f".{settings.MODEL_EXTENSION}", ""))
        for model in models
    ]
    return ModelListResponse(models=cleaned_models)
