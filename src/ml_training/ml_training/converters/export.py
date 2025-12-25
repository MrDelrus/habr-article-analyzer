# export.py

import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict
import json
import zipfile
import tempfile
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType


class ModelExporter:
    
    def __init__(self, model):
        self.model = model
    
    def export(self, save_path: str | Path) -> None:
        save_path = Path(save_path)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            self._export_hub_encoder(tmpdir / "hub_encoder.json")
            self._export_text_encoder(tmpdir / "text_encoder.onnx")
            self._export_predictor(tmpdir / "predictor.onnx")
            self._export_metadata(tmpdir / "metadata.json")
            
            with zipfile.ZipFile(save_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file in tmpdir.glob("*"):
                    zipf.write(file, file.name)

        print(f"Model exported to {save_path}")
        print(f"File size: {save_path.stat().st_size / 1024 / 1024:.2f} MB")

    def _export_hub_encoder(self, save_path: Path) -> None:
        hub_to_vec = self.model.hub_encoder.hub_to_vec

        hub_dict = {
            hub: vec.tolist() 
            for hub, vec in hub_to_vec.items()
        }

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(hub_dict, f, ensure_ascii=False)

        print(f"Hub encoder: {len(hub_dict)} hubs")

    def _export_text_encoder(self, save_path: Path) -> None:
        vectorizer = self.model.text_encoder.vectorizer

        onnx_compatible_pattern = r"[a-zA-Zа-яА-ЯёЁ]{3,}"

        onnx_model = convert_sklearn(
            vectorizer,
            initial_types=[("input", StringTensorType([None, 1]))],
            target_opset={"": 15, "ai.onnx.ml": 2},
            options={id(vectorizer): {"tokenexp": onnx_compatible_pattern}}
        )

        with open(save_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"Text encoder: {len(vectorizer.vocabulary_)} features")
    
    def _export_predictor(self, save_path: Path) -> None:
        first_layer = self.model.predictor.model[0]
        input_dim = first_layer.in_features

        dummy_input = torch.randn(1, input_dim, dtype=torch.float32)

        torch.onnx.export(
            self.model.predictor,
            dummy_input,
            save_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}},
            opset_version=15,
            do_constant_folding=True
        )

        print(f"Predictor: input_dim={input_dim}")

    def _export_metadata(self, save_path: Path) -> None:
        metadata = {
            "text_encoder_dim": self.model.text_encoder.max_features,
            "hub_encoder_dim": self.model.hub_encoder.dim,
            "predictor_input_dim": self.model.predictor.model[0].in_features,
            "version": "1.0"
        }
        
        with open(save_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Metadata saved")


def export_model(model, save_path: str | Path) -> None:
    exporter = ModelExporter(model)
    exporter.export(save_path)
