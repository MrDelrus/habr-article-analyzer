from fastapi import APIRouter, Depends, Query

from backend.auth import get_current_user
from backend.clients import DatabaseClient, get_database_client
from backend.models import User
from core.schemas.api import HistoryResponse

router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=HistoryResponse)
async def get_history(
    limit: int = Query(5, ge=1, le=20),
    database_client: DatabaseClient = Depends(get_database_client),
    current_user: User = Depends(get_current_user),
) -> HistoryResponse:
    history_items = await database_client.fetch_history(current_user.user_id, limit)
    return HistoryResponse(history=history_items)
