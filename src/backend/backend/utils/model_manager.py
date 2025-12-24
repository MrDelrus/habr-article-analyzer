from functools import lru_cache

from backend.utils.onnx_runner import ONNXModelRunner


@lru_cache(maxsize=10)
def get_model_runner(model_key: str) -> ONNXModelRunner:
    return ONNXModelRunner(model_key)
