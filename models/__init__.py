# models/__init__.py
from .base import Base
from .crypto_info import CryptoInfo
from .symbol_trading_rules import SymbolTradingRules
from .users import User
from .pipeline_state import PipelineState, PipelineComponent
from .backfill_progress import BackfillProgress
from .indicator_progress import IndicatorProgress
from .websocket_progress import WebSocketProgress
from .error_log import ErrorLogCurrent, ErrorLogHistory


from .ohlcv_data import (
    Ohlcv1m,
    Ohlcv5m,
    Ohlcv15m,
    Ohlcv1h,
    Ohlcv4h,
    Ohlcv1d,
)

from .indicators import (
    Indicator1m,
    Indicator5m,
    Indicator15m,
    Indicator1h,
    Indicator4h,
    Indicator1d,
)

from .backtesting_features import (
    BacktestingFeatures1m,
    BacktestingFeatures5m,
    BacktestingFeatures15m,
    BacktestingFeatures1h,
    BacktestingFeatures4h,
    BacktestingFeatures1d,
    BACKTESTING_FEATURES_MODELS,
)

# 인터벌별 OHLCV 모델 매핑
OHLCV_MODELS = {
    "1m": Ohlcv1m,
    "5m": Ohlcv5m,
    "15m": Ohlcv15m,
    "1h": Ohlcv1h,
    "4h": Ohlcv4h,
    "1d": Ohlcv1d,
}

# 인터벌별 보조지표 모델 매핑
INDICATOR_MODELS = {
    "1m": Indicator1m,
    "5m": Indicator5m,
    "15m": Indicator15m,
    "1h": Indicator1h,
    "4h": Indicator4h,
    "1d": Indicator1d,
}
