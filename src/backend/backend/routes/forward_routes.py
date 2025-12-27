from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend import get_db, settings
from backend.models import History
from core.schemas.api import ForwardRequest, ForwardResponse

router = APIRouter(prefix="/forward", tags=["forward"])

ML_FORWARD_ENDPOINT = f"{settings.ML_SERVICE_URL}/v0/forward"


@router.post("", response_model=ForwardResponse)
async def forward(
    request: ForwardRequest,
    db: AsyncSession = Depends(get_db),
) -> ForwardResponse:
    query_id = uuid4()
    http_status = status.HTTP_200_OK

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                ML_FORWARD_ENDPOINT,
                headers={"x-internal-key": settings.INTERNAL_API_KEY},
                json=request.model_dump(),
            )

            if resp.status_code != 200:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"ML service error: {resp.text}",
                )

            ml_response = ForwardResponse.model_validate(resp.json())
            if ml_response.result:
                ml_response.result.sort(key=lambda x: x.score, reverse=True)

            return ml_response

    except HTTPException:
        raise

    except Exception as exception:
        http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise HTTPException(
            status_code=http_status,
            detail=f"Error during ML service call: {str(exception)}",
        )

    finally:
        db.add(
            History(
                query_id=query_id,
                endpoint="/forward",
                code_status=http_status,
            )
        )
        await db.commit()
