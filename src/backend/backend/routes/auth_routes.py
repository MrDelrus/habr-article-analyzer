from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from passlib.context import CryptContext

from backend.auth import auth_settings
from backend.clients.database_client import DatabaseClient, get_database_client
from backend.models.user import User
from core.schemas.auth import Token

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bool(pwd_context.verify(plain_password, hashed_password))


def create_access_token(
    data: dict[str, Any], expires_delta: timedelta | None = None
) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=auth_settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode["exp"] = expire
    encoded_jwt: str = jwt.encode(
        to_encode,
        auth_settings.SECRET_KEY,
        algorithm=auth_settings.ALGORITHM,
    )
    return encoded_jwt


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    database_client: DatabaseClient = Depends(get_database_client),
) -> Token:
    user: User | None = await database_client.get_user_by_username(form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    access_token: str = create_access_token({"sub": str(user.user_id)})
    return Token(access_token=access_token, token_type="bearer")
