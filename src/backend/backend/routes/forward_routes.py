from fastapi import APIRouter, HTTPException, status

from backend.config import settings
from backend.utils.model_manager import get_model_runner
from backend.utils.s3_loader import list_models
from core.schemas.api.forward import ForwardRequest, ForwardResponse

router = APIRouter(prefix="/forward", tags=["forward"])


@router.post("", response_model=ForwardResponse)
async def forward(request: ForwardRequest) -> ForwardResponse:
    model_key = f"{request.model_name}.{settings.MODEL_EXTENSION}"

    available_models = list_models()
    if model_key not in available_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{model_key}' is not found",
        )

    try:
        runner = get_model_runner(model_key)
        result = runner.forward(request.input1, request.input2)
    except Exception as exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during model inference: {str(exception)}",
        )

    return ForwardResponse(output=result)
