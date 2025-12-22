import os
from pathlib import Path

import tomli
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_PATH = Path(__file__).resolve().parent / "config.toml"

with open(CONFIG_PATH, "rb") as f:
    config = tomli.load(f)


class Paths(BaseSettings):
    data_dir: Path
    models_dir: Path
    logs_dir: Path


class Settings(BaseSettings):
    log_level: str


class DataSplitSettings(BaseSettings):
    test_size: float
    val_size: float
    random_seed: int


paths = Paths(**{k: PROJECT_ROOT / v for k, v in config["paths"].items()})
settings = Settings(**config["settings"])
data_settings = DataSplitSettings(**config["data_split_settings"])
