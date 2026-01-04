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
from backend.models import History
from core.schemas.api import HistoryItem

DATABASE_URL = settings.DATABASE_URL


class DatabaseClient:
    def __init__(self, url: str) -> None:
        self._url = url
        self._engine: AsyncEngine = create_async_engine(
            url,
            echo=False,
            future=True,
        )

        self._async_session_maker: async_sessionmaker[AsyncSession] = (
            async_sessionmaker(
                bind=self._engine,
                expire_on_commit=False,
            )
        )

    async def fetch_history(self, limit: int = 5) -> list[HistoryItem]:
        if limit > 20:
            raise ValueError("Limit can't be higher 20!")

        query = select(History).order_by(History.timestamp.desc()).limit(limit)
        async with self._async_session_maker() as session:
            result = await session.execute(query)
            rows: list[History] = result.scalars().all()

        return [HistoryItem.model_validate(row) for row in rows]

    async def add_history(
        self,
        *,
        query_id: UUID,
        endpoint: str,
        code_status: int,
    ) -> None:
        history = History(
            query_id=query_id,
            endpoint=endpoint,
            code_status=code_status,
        )

        async with self._async_session_maker() as session:
            session.add(history)
            try:
                await session.commit()
            except Exception:
                await session.rollback()

    async def close(self) -> None:
        await self._engine.dispose()


async def get_database_client() -> AsyncGenerator[DatabaseClient, None]:
    client = DatabaseClient(url=settings.DATABASE_URL)
    try:
        yield client
    finally:
        await client.close()
