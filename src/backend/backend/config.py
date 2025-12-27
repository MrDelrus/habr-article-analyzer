from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    AWS_REGION: str
    DATABASE_URL: str
    ML_SERVICE_URL: str
    INTERNAL_API_KEY: str


settings = Settings()
