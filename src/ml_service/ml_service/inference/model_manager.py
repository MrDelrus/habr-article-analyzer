from functools import lru_cache

from fastapi import HTTPException

from ml_service import settings
from ml_service.client import download_model
from ml_service.inference.interface import BaseHubClassifierInference
from ml_service.inference.onnx_runner import ONNXInference


@lru_cache(maxsize=10)
def get_model_runner(model_name: str) -> BaseHubClassifierInference:
    model_path = f"{model_name}.{settings.MODEL_EXTENSION}"
    if model_name == "dummy":
        s3_model_path = download_model(model_path)
        return ONNXInference(s3_model_path)

    raise HTTPException(f"Model '{model_name}' is not found")
