from pydantic_settings import BaseSettings
from typing import Optional


class FrontendSettings(BaseSettings):
    """Frontend configuration settings"""
    
    api_base_url: str = "http://127.0.0.1:8000"
    default_model_name: str = "BoWDSSM"
    max_file_size_mb: int = 10
    request_timeout: int = 30
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = FrontendSettings()
