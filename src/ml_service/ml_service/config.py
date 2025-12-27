from pydantic_settings import BaseSettings

DEFAULT_HUBS = [
    "closet",
    "itcompanies",
    "infosecurity",
    "programming",
    "webdev",
    "popular_science",
    "javascript",
    "gadgets",
    "finance",
    "business-laws",
]


class Settings(BaseSettings):
    S3_BUCKET_NAME: str
    MODEL_EXTENSION: str
    INTERNAL_API_KEY: str


settings = Settings()
