from fastapi import APIRouter, HTTPException, status

from backend.config import settings
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

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Forward inference is not implemented yet",
    )
