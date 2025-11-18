# models/rest_progress.py
from sqlalchemy import Column, String, TIMESTAMP, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class RestProgress(Base):
    __tablename__ = "rest_progress"
    __table_args__ = {"schema": "trading_data"}

    # PK = run_id + symbol + interval
    run_id = Column(String(64), primary_key=True)
    symbol = Column(String(30), primary_key=True)
    interval = Column(String(10), primary_key=True)

    # PENDING / PROGRESS / SUCCESS / FAILURE
    state = Column(String(20), nullable=False)

    last_candle_ts = Column(TIMESTAMP(timezone=True))
    last_error = Column(Text, nullable=True)

    updated_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
