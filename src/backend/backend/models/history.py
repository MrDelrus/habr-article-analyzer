from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, SmallInteger, String, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.models.base import Base


class History(Base):
    __tablename__ = "history"

    query_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
    )

    endpoint: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    code_status: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
