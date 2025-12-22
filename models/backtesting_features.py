from typing import List, Type
from datetime import datetime
from sqlalchemy import Column, String, TIMESTAMP, Boolean, Numeric, CheckConstraint
from sqlalchemy.orm import Mapped, mapped_column

from models.base import Base

class BacktestingFeaturesBase(Base):
    __abstract__ = True

    symbol: Mapped[str] = mapped_column(String(30), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), primary_key=True)
    
    log_return: Mapped[float] = mapped_column(Numeric(20, 7), nullable=True)
    ema_ratio: Mapped[float] = mapped_column(Numeric(20, 7), nullable=True)
    macd_hist: Mapped[float] = mapped_column(Numeric(20, 7), nullable=True)
    bandwidth: Mapped[float] = mapped_column(Numeric(20, 7), nullable=True)
    pct_b: Mapped[float] = mapped_column(Numeric(20, 7), nullable=True)
    rsi: Mapped[float] = mapped_column(Numeric(20, 7), nullable=True)
    mfi: Mapped[float] = mapped_column(Numeric(20, 7), nullable=True)


class BacktestingFeatures1m(BacktestingFeaturesBase):
    __tablename__ = "backtesting_features_1m"
    __table_args__ = {"schema": "trading_data"}

class BacktestingFeatures5m(BacktestingFeaturesBase):
    __tablename__ = "backtesting_features_5m"
    __table_args__ = {"schema": "trading_data"}

class BacktestingFeatures15m(BacktestingFeaturesBase):
    __tablename__ = "backtesting_features_15m"
    __table_args__ = {"schema": "trading_data"}

class BacktestingFeatures1h(BacktestingFeaturesBase):
    __tablename__ = "backtesting_features_1h"
    __table_args__ = {"schema": "trading_data"}

class BacktestingFeatures4h(BacktestingFeaturesBase):
    __tablename__ = "backtesting_features_4h"
    __table_args__ = {"schema": "trading_data"}

class BacktestingFeatures1d(BacktestingFeaturesBase):
    __tablename__ = "backtesting_features_1d"
    __table_args__ = {"schema": "trading_data"}


BACKTESTING_FEATURES_MODELS = {
    "1m": BacktestingFeatures1m,
    "5m": BacktestingFeatures5m,
    "15m": BacktestingFeatures15m,
    "1h": BacktestingFeatures1h,
    "4h": BacktestingFeatures4h,
    "1d": BacktestingFeatures1d,
}
