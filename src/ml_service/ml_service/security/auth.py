from fastapi import Header, HTTPException

from ml_service import settings


def verify_internal_key(
    x_internal_key: str = Header(..., alias="x-internal-key"),
) -> None:
    if x_internal_key != settings.INTERNAL_API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid internal API key",
        )
