from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    S3_BUCKET_NAME: str
    MODEL_EXTENSION: str
    AWS_REGION: str
    DATABASE_URL: str


settings = Settings()
