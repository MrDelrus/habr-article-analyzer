from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    S3_BUCKET_NAME: str
    MODEL_EXTENSION: str
    AWS_REGION: str


settings = Settings()
