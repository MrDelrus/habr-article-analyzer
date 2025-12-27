from typing import List, Optional

from pydantic import BaseModel, Field


class ForwardRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model.")
    text: str = Field(..., description="Article from habr.")
    hubs: Optional[List[str]] = Field(
        None, description="List of possible hubs. If `None`, default list is used."
    )


class HubScore(BaseModel):
    hub: str
    score: float


class ForwardResponse(BaseModel):
    result: Optional[List[HubScore]] = Field(None, description="List of predictions.")
    error: Optional[str] = Field(None, description="Error description.")
