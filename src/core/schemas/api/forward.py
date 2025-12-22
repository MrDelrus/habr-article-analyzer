from typing import List, Optional

from pydantic import BaseModel


class ForwardRequest(BaseModel):
    model_name: str
    text: str
    hubs: Optional[List[str]] = None


class HubScore(BaseModel):
    hub: str
    score: float


class ForwardResponse(BaseModel):
    result: Optional[List[HubScore]] = None
    error: Optional[str] = None
