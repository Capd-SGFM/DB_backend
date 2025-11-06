from datetime import datetime

from sqlalchemy import String, TIMESTAMP, Numeric, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base


class _IndicatorBase(Base):
    """보조지표 공통 컬럼"""

    __abstract__ = True
    __table_args__ = {"schema": "trading_data"}

    symbol: Mapped[str] = mapped_column(
        String(30),
        ForeignKey("metadata.crypto_info.symbol", ondelete="CASCADE"),
        primary_key=True,
    )
    timestamp: Mapped[datetime] = mapped_column(
        "timestamp",
        TIMESTAMP(timezone=True),
        primary_key=True,
    )

    rsi_14: Mapped[float] = mapped_column(Numeric(20, 7), nullable=False)
    ema_7: Mapped[float] = mapped_column(Numeric(20, 7), nullable=False)
    ema_21: Mapped[float] = mapped_column(Numeric(20, 7), nullable=False)
    ema_99: Mapped[float] = mapped_column(Numeric(20, 7), nullable=False)
    sma_7: Mapped[float] = mapped_column(Numeric(20, 7), nullable=False)
    sma_21: Mapped[float] = mapped_column(Numeric(20, 7), nullable=False)
    sma_99: Mapped[float] = mapped_column(Numeric(20, 7), nullable=False)
    macd: Mapped[float] = mapped_column(Numeric(20, 7), nullable=False)
    macd_signal: Mapped[float] = mapped_column(Numeric(20, 7), nullable=False)
    macd_hist: Mapped[float] = mapped_column(Numeric(20, 7), nullable=False)
    bb_upper: Mapped[float] = mapped_column(Numeric(20, 7), nullable=False)
    bb_middle: Mapped[float] = mapped_column(Numeric(20, 7), nullable=False)
    bb_lower: Mapped[float] = mapped_column(Numeric(20, 7), nullable=False)
    volume_20: Mapped[float] = mapped_column(Numeric(20, 3), nullable=False)

    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


# --- 개별 타임프레임 테이블 ---
class Indicator1m(_IndicatorBase):
    __tablename__ = "indicators_1m"


class Indicator3m(_IndicatorBase):
    __tablename__ = "indicators_3m"


class Indicator5m(_IndicatorBase):
    __tablename__ = "indicators_5m"


class Indicator15m(_IndicatorBase):
    __tablename__ = "indicators_15m"


class Indicator30m(_IndicatorBase):
    __tablename__ = "indicators_30m"


class Indicator1h(_IndicatorBase):
    __tablename__ = "indicators_1h"


class Indicator4h(_IndicatorBase):
    __tablename__ = "indicators_4h"


class Indicator1d(_IndicatorBase):
    __tablename__ = "indicators_1d"


class Indicator1w(_IndicatorBase):
    __tablename__ = "indicators_1w"


class Indicator1M(_IndicatorBase):
    __tablename__ = "indicators_1M"
