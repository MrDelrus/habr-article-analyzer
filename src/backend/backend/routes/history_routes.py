from fastapi import APIRouter, Depends, Query

from backend.clients import DatabaseClient, get_database_client
from core.schemas.api import HistoryResponse

router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=HistoryResponse)
async def get_history(
    limit: int = Query(5, ge=1, le=20),
    database_client: DatabaseClient = Depends(get_database_client),
) -> HistoryResponse:
    history_items = database_client.fetch_history(limit)
    return HistoryResponse(history=history_items)
