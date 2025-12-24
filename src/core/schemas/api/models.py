from typing import List

from pydantic import BaseModel


class ModelItem(BaseModel):
    name: str


class ModelListResponse(BaseModel):
    models: List[ModelItem]
