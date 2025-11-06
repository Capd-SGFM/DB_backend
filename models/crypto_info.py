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

    # 3. 정밀도
    price_precision: Mapped[Optional[int]] = mapped_column(Integer)
    quantity_precision: Mapped[Optional[int]] = mapped_column(Integer)

    # 4. 증거금 및 수수료
    required_margin_percent: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 5))
    maint_margin_percent: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 5))
    liquidation_fee: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 5))

    # 6. PRICE_FILTER
    tick_size: Mapped[Optional[Decimal]] = mapped_column(Numeric(30, 15))

    # 7. LOT_SIZE (지정가 주문 수량 제한)
    min_qty: Mapped[Optional[Decimal]] = mapped_column(Numeric(30, 15))
    max_qty: Mapped[Optional[Decimal]] = mapped_column(Numeric(30, 15))
    step_size: Mapped[Optional[Decimal]] = mapped_column(Numeric(30, 15))

    # 8. MARKET_LOT_SIZE (시장가 주문 수량 제한)
    market_min_qty: Mapped[Optional[Decimal]] = mapped_column(Numeric(30, 15))
    market_max_qty: Mapped[Optional[Decimal]] = mapped_column(Numeric(30, 15))
    market_step_size: Mapped[Optional[Decimal]] = mapped_column(Numeric(30, 15))

    # 9. 최소 주문 금액
    min_notional: Mapped[Optional[Decimal]] = mapped_column(Numeric(30, 15))

    # 10. 최대 미체결 주문 수
    max_num_orders: Mapped[Optional[int]] = mapped_column(Integer)

    def __repr__(self) -> str:
        return (
            f"CryptoInfo(symbol={self.symbol!r}, pair={self.pair!r}, "
            f"price_precision={self.price_precision!r}, quantity_precision={self.quantity_precision!r})"
        )
