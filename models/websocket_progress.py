# models/websocket_progress.py
from __future__ import annotations

from datetime import datetime
from sqlalchemy import String, Integer, TIMESTAMP, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .base import Base


class WebSocketProgress(Base):
    __tablename__ = "websocket_progress"
    __table_args__ = {"schema": "trading_data"}

    # PK = run_id + symbol + interval
    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    symbol: Mapped[str] = mapped_column(
        String(30),
        ForeignKey("metadata.crypto_info.symbol", ondelete="CASCADE"),
        primary_key=True,
    )
    interval: Mapped[str] = mapped_column(String(10), primary_key=True)

    # CONNECTED / DISCONNECTED / ERROR
    state: Mapped[str] = mapped_column(String(20), nullable=False)

    # 마지막 메시지 수신 시각
    last_message_ts: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    # 수신된 메시지 개수
    message_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # 에러 메시지
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 업데이트 시각
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


def reset_websocket_progress():
    """WebSocket 진행률 전체 초기화 (TRUNCATE)"""
    from db_module.connect_sqlalchemy_engine import SyncSessionLocal
    from sqlalchemy import text

    with SyncSessionLocal() as session, session.begin():
        session.execute(text("TRUNCATE TABLE trading_data.websocket_progress"))
