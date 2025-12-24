import numpy as np
import onnxruntime as ort

from backend.utils.s3_loader import download_model


class ONNXModelRunner:
    def __init__(self, s3_key: str):
        local_path = download_model(s3_key)
        self.session = ort.InferenceSession(local_path)

    def _str_to_tensor(self, s: str, max_len: int = 10) -> np.ndarray:
        arr = np.zeros((1, max_len), dtype=np.int64)
        for i, c in enumerate(s[:max_len]):
            arr[0, i] = ord(c)
        return arr

    def forward(self, text: str, hub: str) -> float:
        str1_tensor = self._str_to_tensor(text)
        str2_tensor = self._str_to_tensor(hub)

        inputs = {"text": str1_tensor, "hub": str2_tensor}
        output = self.session.run(None, inputs)

        return float(output[0][0])
