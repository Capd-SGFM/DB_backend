# models/indicator_progress.py
from datetime import datetime

from sqlalchemy import Column, String, Numeric, DateTime, Text
from sqlalchemy.sql import func

from .base import Base


class IndicatorProgress(Base):
    __tablename__ = "indicator_progress"
    __table_args__ = {"schema": "trading_data"}

    run_id = Column(String(64), primary_key=True)
    symbol = Column(String(30), primary_key=True)
    interval = Column(String(10), primary_key=True)

    state = Column(String(20), nullable=False)  # PENDING / PROGRESS / SUCCESS / FAILURE
    pct_time = Column(Numeric(5, 2), nullable=False, default=0)

    last_candle_ts = Column(DateTime(timezone=True), nullable=True)
    last_error = Column(Text, nullable=True)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


def reset_indicator_progress():
    """새로운 run 시작 전에 진행현황 초기화용."""
    from db_module.connect_sqlalchemy_engine import SyncSessionLocal

    with SyncSessionLocal() as session, session.begin():
        session.execute("TRUNCATE trading_data.indicator_progress")
