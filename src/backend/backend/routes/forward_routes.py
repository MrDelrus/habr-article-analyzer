from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.db import get_db
from backend.models.history import History
from backend.utils.model_manager import get_model_runner
from backend.utils.s3_loader import list_models
from core.schemas.api.forward import ForwardRequest, ForwardResponse, HubScore

router = APIRouter(prefix="/forward", tags=["forward"])


DEFAULT_HUBS = [
    "closet",
    "itcompanies",
    "infosecurity",
    "programming",
    "webdev",
    "popular_science",
    "javascript",
    "gadgets",
    "finance",
    "business-laws",
]


@router.post("", response_model=ForwardResponse)
async def forward(
    request: ForwardRequest,
    db: AsyncSession = Depends(get_db),
) -> ForwardResponse:
    query_id = uuid4()
    http_status = status.HTTP_200_OK

    model_key = f"{request.model_name}.{settings.MODEL_EXTENSION}"

    try:
        available_models = list_models()
        if model_key not in available_models:
            http_status = status.HTTP_400_BAD_REQUEST
            raise HTTPException(
                status_code=http_status,
                detail=f"Model '{model_key}' is not found",
            )

        hubs_to_score = request.hubs or DEFAULT_HUBS

        runner = get_model_runner(model_key)
        scores = []
        for hub in hubs_to_score:
            score = runner.forward(request.text, hub)
            scores.append(HubScore(hub=hub, score=score))

        scores.sort(key=lambda x: x.score, reverse=True)

        return ForwardResponse(result=scores)

    except HTTPException:
        raise

    except Exception as exception:
        http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise HTTPException(
            status_code=http_status,
            detail=f"Error during model inference: {str(exception)}",
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
