from typing import Optional
from decimal import Decimal
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer, Numeric
from .base import Base


class CryptoInfo(Base):
    """거래 가능한 심볼 및 기본 규칙 정보 (metadata.crypto_info)"""

    __tablename__ = "crypto_info"
    __table_args__ = {"schema": "metadata"}

    # 1. 심볼 (예: 'BTC', 'ETH', ...)
    symbol: Mapped[str] = mapped_column(String(30), primary_key=True)

    # 2. 실제 API 페어 (예: 'BTCUSDT')
    pair: Mapped[str] = mapped_column(String(30), nullable=False, unique=True)

    def __repr__(self) -> str:
        return f"CryptoInfo(symbol={self.symbol!r}, pair={self.pair!r})"
