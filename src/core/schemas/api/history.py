from datetime import datetime
from typing import List
from uuid import UUID

from pydantic import BaseModel


class HistoryItem(BaseModel):
    query_id: UUID
    endpoint: str
    code_status: int
    timestamp: datetime

    class Config:
        orm_mode = True


class HistoryResponse(BaseModel):
    history: List[HistoryItem]
