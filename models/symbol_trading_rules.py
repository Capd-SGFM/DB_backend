from typing import Optional
from decimal import Decimal
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer, Numeric, Boolean, DateTime
from sqlalchemy.sql import func
from .base import Base

class SymbolTradingRules(Base):
    """종목별 선물 거래 조건 (futures.symbol_trading_rules)"""

    __tablename__ = "symbol_trading_rules"
    __table_args__ = {"schema": "futures"}

    # 1. 심볼 (FK to metadata.crypto_info)
    symbol: Mapped[str] = mapped_column(String(30), primary_key=True)

    # 2. 소수점 정밀도
    price_precision: Mapped[int] = mapped_column(Integer, nullable=False)
    quantity_precision: Mapped[int] = mapped_column(Integer, nullable=False)

    # 3. 가격 관련
    tick_size: Mapped[Decimal] = mapped_column(Numeric(30, 15), nullable=False)

    # 4. 수량 제한 (지정가)
    min_qty: Mapped[Decimal] = mapped_column(Numeric(30, 15), nullable=False)
    max_qty: Mapped[Decimal] = mapped_column(Numeric(30, 15), nullable=False)
    step_size: Mapped[Decimal] = mapped_column(Numeric(30, 15), nullable=False)

    # 5. 수량 제한 (시장가)
    market_min_qty: Mapped[Decimal] = mapped_column(Numeric(30, 15), nullable=False)
    market_max_qty: Mapped[Decimal] = mapped_column(Numeric(30, 15), nullable=False)
    market_step_size: Mapped[Decimal] = mapped_column(Numeric(30, 15), nullable=False)

    # 6. 금액 제한
    min_notional: Mapped[Decimal] = mapped_column(Numeric(30, 15), nullable=False)

    # 7. 주문 개수 제한
    max_num_orders: Mapped[int] = mapped_column(Integer, default=200, nullable=False)
    max_num_algo_orders: Mapped[int] = mapped_column(Integer, default=10, nullable=False)

    # 8. 증거금 및 레버리지
    required_margin_percent: Mapped[Decimal] = mapped_column(Numeric(10, 5), nullable=False)
    maint_margin_percent: Mapped[Decimal] = mapped_column(Numeric(10, 5), nullable=False)
    max_leverage: Mapped[int] = mapped_column(Integer, default=125, nullable=False)

    # 9. 수수료
    maker_commission_rate: Mapped[Decimal] = mapped_column(Numeric(10, 5), default=0.0002, nullable=False)
    taker_commission_rate: Mapped[Decimal] = mapped_column(Numeric(10, 5), default=0.0004, nullable=False)
    liquidation_fee: Mapped[Decimal] = mapped_column(Numeric(10, 5), default=0.005, nullable=False)

    # 10. 메타 정보
    is_trading_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self) -> str:
        return (
            f"SymbolTradingRules(symbol={self.symbol!r}, "
            f"price_precision={self.price_precision!r}, quantity_precision={self.quantity_precision!r})"
        )
