from .base import Base

from .crypto_info import CryptoInfo
from .users import User
from .ohlcv_data import (
    Ohlcv1m,
    Ohlcv3m,
    Ohlcv5m,
    Ohlcv15m,
    Ohlcv30m,
    Ohlcv1h,
    Ohlcv4h,
    Ohlcv1d,
    Ohlcv1w,
    Ohlcv1M,
)
from .indicators import (
    Indicator1m,
    Indicator3m,
    Indicator5m,
    Indicator15m,
    Indicator30m,
    Indicator1h,
    Indicator4h,
    Indicator1d,
    Indicator1w,
    Indicator1M,
)


# 모든 OHLCV 모델을 인터벌 문자열로 쉽게 찾을 수 있도록 딕셔너리 만듦
OHLCV_MODELS = {
    # "1m": Ohlcv1m,
    # "3m": Ohlcv3m,
    # "5m": Ohlcv5m,
    # "15m": Ohlcv15m,
    # "30m": Ohlcv30m,
    # "1h": Ohlcv1h,
    "4h": Ohlcv4h,
    "1d": Ohlcv1d,
    # "1w": Ohlcv1w,
    # "1M": Ohlcv1M,
}

# 모든 보조지표 모델 딕셔너리
INDICATOR_MODELS = {
    # "1m": Indicator1m,
    # "3m": Indicator3m,
    # "5m": Indicator5m,
    # "15m": Indicator15m,
    # "30m": Indicator30m,
    # "1h": Indicator1h,
    "4h": Indicator4h,
    "1d": Indicator1d,
    # "1w": Indicator1w,
    # "1M": Indicator1M,
}
