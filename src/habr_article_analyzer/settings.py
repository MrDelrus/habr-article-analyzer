from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = project_root / "data"
    models_dir: Path = project_root / "models"

    log_level: str = "INFO"


settings = Settings()
