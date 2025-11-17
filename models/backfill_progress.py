# models/backfill_progress.py
from __future__ import annotations

from datetime import datetime
from sqlalchemy import String, Numeric, TIMESTAMP, Text, text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from db_module.connect_sqlalchemy_engine import SyncSessionLocal
from .base import Base


class BackfillProgress(Base):
    __tablename__ = "backfill_progress"
    __table_args__ = {"schema": "trading_data"}

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    symbol: Mapped[str] = mapped_column(
        String(30),
        ForeignKey("metadata.crypto_info.symbol", ondelete="CASCADE"),
        primary_key=True,
    )
    interval: Mapped[str] = mapped_column(String(10), primary_key=True)

    state: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'PENDING'")
    )
    pct_time: Mapped[float] = mapped_column(
        Numeric(5, 2), nullable=False, server_default=text("0")
    )
    last_candle_ts: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("now()"),
        onupdate=text("now()"),
    )


def reset_backfill_progress():
    """Backfill 진행률 전체 초기화"""
    with SyncSessionLocal() as session, session.begin():
        session.execute(text("TRUNCATE TABLE trading_data.backfill_progress"))
