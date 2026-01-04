from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status

from backend.clients import (
    DatabaseClient,
    MLServiceClient,
    get_database_client,
    get_ml_service_client,
)
from core.schemas.api import ForwardRequest, ForwardResponse

router = APIRouter(prefix="/forward", tags=["forward"])


@router.post("", response_model=ForwardResponse)
async def forward(
    request: ForwardRequest,
    ml_service_client: MLServiceClient = Depends(get_ml_service_client),
    db: DatabaseClient = Depends(get_database_client),
) -> ForwardResponse:
    query_id = uuid4()
    http_status = status.HTTP_200_OK

    try:
        response: ForwardResponse = await ml_service_client.forward(request)
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
            detail=f"ML Service error: {str(exception)}",
        )

    finally:
        try:
            await db.add_history(
                query_id=query_id, endpoint="/forward", status=http_status
            )
        except Exception:
            pass
