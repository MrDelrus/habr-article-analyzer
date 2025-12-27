from datetime import datetime
from typing import List
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class HistoryItem(BaseModel):
    query_id: UUID
    endpoint: str
    code_status: int
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)


class HistoryResponse(BaseModel):
    history: List[HistoryItem] = Field(..., description="List of last queries.")
