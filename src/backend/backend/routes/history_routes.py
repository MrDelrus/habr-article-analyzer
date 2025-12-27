from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend import get_db
from backend.models import History
from core.schemas.api import HistoryItem, HistoryResponse

router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=HistoryResponse)
async def get_history(
    limit: int = Query(5, ge=1, le=20),
    db: AsyncSession = Depends(get_db),
) -> HistoryResponse:
    stmt = select(History).order_by(History.timestamp.desc()).limit(limit)
    result = await db.execute(stmt)
    rows: list[History] = result.scalars().all()

    history_items: list[HistoryItem] = [HistoryItem.from_orm(row) for row in rows]

    return HistoryResponse(history=history_items)
