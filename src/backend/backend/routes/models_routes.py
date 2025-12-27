from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend import get_db, settings
from backend.models import History
from core.schemas.api import ModelListResponse

router = APIRouter(prefix="/models", tags=["models"])

ML_MODELS_ENDPOINT = f"{settings.ML_SERVICE_URL}/v0/models"


@router.get("", response_model=ModelListResponse)
async def get_models(db: AsyncSession = Depends(get_db)) -> ModelListResponse:
    query_id = uuid4()
    http_status = status.HTTP_200_OK

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                ML_MODELS_ENDPOINT,
                headers={"x-internal-key": settings.INTERNAL_API_KEY},
            )

            if resp.status_code != 200:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"ML service error: {resp.text}",
                )

            ml_response = ModelListResponse.model_validate(resp.json())
            return ml_response

    finally:
        db.add(
            History(
                query_id=query_id,
                endpoint="/models",
                code_status=http_status,
            )
        )
        await db.commit()
