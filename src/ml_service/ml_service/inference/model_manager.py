from functools import lru_cache

from fastapi import HTTPException

from ml_service import settings
from ml_service.client import download_model
from ml_service.inference.interface import BaseHubClassifierInference
from ml_service.inference.onnx_runner import ONNXInference


@lru_cache(maxsize=10)
def get_model_runner(model_name: str) -> BaseHubClassifierInference:
    model_file = f"{model_name}.{settings.MODEL_EXTENSION}"
    model_path = download_model(model_file)
    return ONNXInference(model_path)

    raise HTTPException(f"Model '{model_name}' is not found")
