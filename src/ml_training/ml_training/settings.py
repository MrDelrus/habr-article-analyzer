from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project_root: Path = Path(__file__).resolve().parent.parent.parent.parent
    data_dir: Path = project_root / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    models_dir: Path = project_root / "models"
    logs_dir: Path = project_root / "logs"

    log_level: str = "INFO"


class DataSettings(BaseSettings):
    batch_size: int = 50_000
    test_size: float = 0.2
    val_size: float = 0.2
    random_seed: int = 42

    top_hubs_count: int = 50
    max_positives: int = 5
    num_negatives: int = 5


settings = Settings()
data_settings = DataSettings()
