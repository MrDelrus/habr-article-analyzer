from fastapi import APIRouter, HTTPException, status

from backend.config import settings
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
async def forward(request: ForwardRequest) -> ForwardResponse:
    model_key = f"{request.model_name}.{settings.MODEL_EXTENSION}"

    available_models = list_models()
    if model_key not in available_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{model_key}' is not found",
        )

    hubs_to_score = request.hubs or DEFAULT_HUBS

    try:
        runner = get_model_runner(model_key)
        scores = []
        for hub in hubs_to_score:
            score = runner.forward(request.text, hub)
            scores.append(HubScore(hub=hub, score=score))

        scores.sort(key=lambda x: x.score, reverse=True)

    except Exception as exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during model inference: {str(exception)}",
        )

    return ForwardResponse(result=scores)
