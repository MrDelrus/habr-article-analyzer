from functools import lru_cache

from backend.utils.onnx_runner import ONNXInference


@lru_cache(maxsize=10)
def get_model_runner(model_key: str) -> ONNXInference:
    return ONNXInference(model_key)
