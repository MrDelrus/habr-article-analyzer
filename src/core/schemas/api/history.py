from datetime import datetime
from typing import List

from pydantic import BaseModel


class HistoryItem(BaseModel):
    username: str
    query_name: str
    code_name: str
    timestamp: datetime


class HistoryResponse(BaseModel):
    history: List[HistoryItem]
