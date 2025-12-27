from fastapi import APIRouter, Depends, HTTPException, status

from core.schemas.api import ForwardRequest, ForwardResponse, HubScore
from ml_service import DEFAULT_HUBS
from ml_service.inference import get_model_runner
from ml_service.security import verify_internal_key

router = APIRouter(
    prefix="/v0/forward",
    tags=["forward"],
    dependencies=[Depends(verify_internal_key)],
)


@router.post("", response_model=ForwardResponse)
async def forward(request: ForwardRequest) -> ForwardResponse:
    try:
        hubs_to_score = request.hubs or DEFAULT_HUBS

        runner = get_model_runner(request.model_name)
        scores = []
        for hub in hubs_to_score:
            score = runner.predict_proba(request.text, hub)
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
