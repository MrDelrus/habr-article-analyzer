from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status

from backend import settings
from backend.clients import (
    DatabaseClient,
    MLServiceClient,
    get_database_client,
    get_ml_service_client,
)
from core.schemas.api import ModelListResponse

router = APIRouter(prefix="/models", tags=["models"])

ML_MODELS_ENDPOINT = f"{settings.ML_SERVICE_URL}/v0/models"


@router.get("", response_model=ModelListResponse)
async def get_models(
    ml_service_client: MLServiceClient = Depends(get_ml_service_client),
    db: DatabaseClient = Depends(get_database_client),
) -> ModelListResponse:
    query_id = uuid4()
    http_status = 200

    try:
        response: ModelListResponse = await ml_service_client.models()
        return response

    except HTTPException as exception:
        http_status = exception.status_code
        raise HTTPException(
            status_code=http_status, detail=f"ML Service error: {str(exception)}"
        )

    except Exception as exception:
        http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise HTTPException(
            status_code=http_status,
            detail=f"ML service error: {str(exception)}",
        )

    finally:
        try:
            await db.add_history(
                query_id=query_id, endpoint="/models", status=http_status
            )
        except Exception:
            pass
