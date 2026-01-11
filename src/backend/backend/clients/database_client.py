from collections.abc import AsyncGenerator
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend import settings
from backend.models import History, User
from core.schemas.api import HistoryItem

DATABASE_URL = settings.DATABASE_URL

engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
)

_session_maker: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
)


class DatabaseClient:
    def __init__(
        self, session_maker: async_sessionmaker[AsyncSession], url: str
    ) -> None:
        self._url = url
        self._async_session_maker = session_maker

    async def fetch_history(self, user_id: str, limit: int = 5) -> list[HistoryItem]:
        if limit > 20:
            raise ValueError("Limit can't be higher 20!")

        query = (
            select(History)
            .where(History.user_id == user_id)
            .order_by(History.timestamp.desc())
            .limit(limit)
        )

        async with self._async_session_maker() as session:
            result = await session.execute(query)
            rows: list[History] = result.scalars().all()

        return [HistoryItem.model_validate(row) for row in rows]

    async def add_history(
        self, *, query_id: UUID, endpoint: str, code_status: int, user_id: str
    ) -> None:
        history = History(
            query_id=query_id,
            endpoint=endpoint,
            code_status=code_status,
            user_id=user_id,
        )

        async with self._async_session_maker() as session:
            session.add(history)
            try:
                await session.commit()
            except Exception:
                await session.rollback()

    async def get_user_by_id(self, user_id: str) -> User | None:
        async with self._async_session_maker() as session:
            result = await session.execute(select(User).where(User.user_id == user_id))
            return result.scalars().first()

    async def get_user_by_username(self, username: str) -> User | None:
        async with self._async_session_maker() as session:
            result = await session.execute(
                select(User).where(User.username == username)
            )
            return result.scalars().first()


async def get_database_client() -> AsyncGenerator[DatabaseClient, None]:
    client = DatabaseClient(session_maker=_session_maker, url=DATABASE_URL)
    yield client
