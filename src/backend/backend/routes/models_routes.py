from uuid import uuid4

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.db import get_db
from backend.models.history import History
from backend.utils.s3_loader import list_models
from core.schemas.api.models import ModelItem, ModelListResponse

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelListResponse)
async def get_models(
    db: AsyncSession = Depends(get_db),
) -> ModelListResponse:
    query_id = uuid4()
    http_status = status.HTTP_200_OK

    try:
        models = list_models()
        cleaned_models = [
            ModelItem(name=model.replace(f".{settings.MODEL_EXTENSION}", ""))
            for model in models
        ]
        return ModelListResponse(models=cleaned_models)

    finally:
        db.add(
            History(
                query_id=query_id,
                endpoint="/models",
                code_status=http_status,
            )
        )
        await db.commit()
