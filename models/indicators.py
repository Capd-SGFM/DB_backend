from datetime import datetime

from sqlalchemy import String, TIMESTAMP, Numeric, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class _IndicatorBase(Base):
    """보조지표 공통 컬럼 (schema: trading_data)"""

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

    # === 여기서부터는 02_trading_data_schema.sql과 1:1 매칭 ===
    rsi_14: Mapped[float] = mapped_column(Numeric(20, 7), nullable=False)
    ema_7: Mapped[float] = mapped_column(Numeric(20, 7), nullable=False)
    ema_21: Mapped[float] = mapped_column(Numeric(20, 7), nullable=False)
    ema_99: Mapped[float] = mapped_column(Numeric(20, 7), nullable=True)

    macd: Mapped[float] = mapped_column(Numeric(20, 7), nullable=False)
    macd_signal: Mapped[float] = mapped_column(Numeric(20, 7), nullable=False)
    macd_hist: Mapped[float] = mapped_column(Numeric(20, 7), nullable=False)

    bb_upper: Mapped[float] = mapped_column(Numeric(20, 7), nullable=False)
    bb_middle: Mapped[float] = mapped_column(Numeric(20, 7), nullable=False)
    bb_lower: Mapped[float] = mapped_column(Numeric(20, 7), nullable=False)

    volume_20: Mapped[float] = mapped_column(Numeric(20, 3), nullable=False)


# --- 개별 타임프레임 테이블 ---
class Indicator1m(_IndicatorBase):
    __tablename__ = "indicators_1m"



class Indicator5m(_IndicatorBase):
    __tablename__ = "indicators_5m"


class Indicator15m(_IndicatorBase):
    __tablename__ = "indicators_15m"


class Indicator1h(_IndicatorBase):
    __tablename__ = "indicators_1h"


class Indicator4h(_IndicatorBase):
    __tablename__ = "indicators_4h"


class Indicator1d(_IndicatorBase):
    __tablename__ = "indicators_1d"

