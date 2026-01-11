from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    S3_BUCKET_NAME: str
    MODEL_EXTENSION: str
    INTERNAL_API_KEY: str
    AWS_REGION: str


settings = Settings()
