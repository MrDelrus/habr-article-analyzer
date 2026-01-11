from typing import List

from pydantic import BaseModel, Field


class ModelItem(BaseModel):
    name: str


class ModelListResponse(BaseModel):
    models: List[ModelItem] = Field(..., description="List of available models")
