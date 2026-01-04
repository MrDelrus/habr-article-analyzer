from pydantic_settings import BaseSettings


class AuthSettings(BaseSettings):
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    class Config:
        env_prefix = "AUTH_"
        env_file = ".env"


auth_settings = AuthSettings()
