from sqlalchemy import Column, Integer, String, Text, DateTime, func
from .base import Base

class ErrorLogCurrent(Base):
    """현재 파이프라인 세션에서 발생한 에러 로그"""
    __tablename__ = "error_logs_current"
    __table_args__ = {"schema": "trading_data"}

    id = Column(Integer, primary_key=True, index=True)
    component = Column(String(50), nullable=False)  # WEBSOCKET, BACKFILL, etc.
    symbol = Column(String(20), nullable=True)
    interval = Column(String(10), nullable=True)
    error_message = Column(Text, nullable=False)
    occurred_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

class ErrorLogHistory(Base):
    """파이프라인 종료 시 보관된 과거 에러 로그"""
    __tablename__ = "error_logs_history"
    __table_args__ = {"schema": "trading_data"}

    id = Column(Integer, primary_key=True, index=True)
    component = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=True)
    interval = Column(String(10), nullable=True)
    error_message = Column(Text, nullable=False)
    occurred_at = Column(DateTime(timezone=True), nullable=False)
    archived_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
