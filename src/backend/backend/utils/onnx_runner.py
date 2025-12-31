import json
import tempfile
import zipfile
from pathlib import Path
from typing import Dict

import numpy as np
import onnxruntime as ort

from backend.utils.s3_loader import download_model


class ONNXInference:

    def __init__(self, model_key: str | Path):
        self.model_path = download_model(model_key)

        self._tmpdir = tempfile.TemporaryDirectory()
        self._extract_model()

        self._load_hub_encoder()
        self._load_text_encoder()
        self._load_predictor()
        self._load_metadata()

    def _extract_model(self) -> None:
        with zipfile.ZipFile(self.model_path, "r") as zipf:
            zipf.extractall(self._tmpdir.name)

        self._tmpdir_path = Path(self._tmpdir.name)

    def _load_hub_encoder(self) -> None:
        hub_path = self._tmpdir_path / "hub_encoder.json"

        with open(hub_path, "r", encoding="utf-8") as f:
            hub_dict = json.load(f)

        self.hub_to_vec: Dict[str, np.ndarray] = {
            hub: np.array(vec, dtype=np.float32) for hub, vec in hub_dict.items()
        }

        if self.hub_to_vec:
            first_vec = next(iter(self.hub_to_vec.values()))
            self.hub_dim = len(first_vec)
            self.default_hub_vec = np.zeros(self.hub_dim, dtype=np.float32)
        else:
            self.hub_dim = 0
            self.default_hub_vec = np.array([], dtype=np.float32)

    def _load_text_encoder(self) -> None:
        text_encoder_path = self._tmpdir_path / "text_encoder.onnx"

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.text_encoder_session = ort.InferenceSession(
            str(text_encoder_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

    def _load_predictor(self) -> None:
        predictor_path = self._tmpdir_path / "predictor.onnx"

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.predictor_session = ort.InferenceSession(
            str(predictor_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

    def _load_metadata(self) -> None:
        metadata_path = self._tmpdir_path / "metadata.json"

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

    def encode_text(self, text: str) -> np.ndarray:
        text_input = np.array([[text]], dtype=object)
        result = self.text_encoder_session.run(None, {"input": text_input})
        return result[0].astype(np.float32)

    def encode_hub(self, hub: str) -> np.ndarray:
        if hub in self.hub_to_vec:
            return self.hub_to_vec[hub].reshape(1, -1)
        else:
            return self.default_hub_vec.reshape(1, -1)

    def predict_proba(self, text: str, hub: str) -> float:
        text_vec = self.encode_text(text)
        hub_vec = self.encode_hub(hub)
        combined = np.concatenate([text_vec, hub_vec], axis=1)
        result = self.predictor_session.run(None, {"input": combined})
        return float(result[0][0])

    def predict(self, text: str, hub: str) -> int:
        proba = self.predict_proba(text, hub)
        return int(proba > 0.5)

    def batch_predict_proba(self, texts: list[str], hubs: list[str]) -> np.ndarray:
        assert len(texts) == len(hubs), "texts and hubs must have same length"

        results = []
        for text, hub in zip(texts, hubs):
            results.append(self.predict_proba(text, hub))

        return np.array(results)

    def __del__(self) -> None:
        if hasattr(self, "_tmpdir"):
            self._tmpdir.cleanup()
