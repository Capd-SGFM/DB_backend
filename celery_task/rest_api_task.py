import httpx
import pandas as pd
import pandas_ta as ta
import time
from celery import Task
from datetime import datetime, timezone
from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert
from loguru import logger

from db_module.connect_sqlalchemy_engine import SyncSessionLocal
from models import OHLCV_MODELS, INDICATOR_MODELS
from . import celery_app

# 바이낸스 API 설정
BINANCE_FAPI_URL = "https://fapi.binance.com/fapi/v1/klines"
KLINE_LIMIT = 1000
KLINE_REQUEST_WEIGHT = 5

# Rate Limit 제어용 상수
RATE_LIMIT_HEADER = "x-mbx-used-weight-1m"
MAX_WEIGHT_PER_MINUTE = 2400
SAFETY_MARGIN_PERCENT = 0.8
TARGET_WEIGHT = MAX_WEIGHT_PER_MINUTE * SAFETY_MARGIN_PERCENT

INTERVAL_TO_MS = {
    "1m": 60000,
    "3m": 180000,
    "5m": 300000,
    "15m": 900000,
    "30m": 1800000,
    "1h": 3600000,
    "4h": 14400000,
    "1d": 86400000,
    "1w": 7 * 86400000,
    "1M": 30 * 86400000,
}


def get_start_time_ms(symbol: str, interval: str, OhlcvModel) -> int | None:
    """[수정] 동기식으로 DB에서 마지막 캔들 시간을 찾습니다."""
    with SyncSessionLocal() as session:
        stmt = select(func.max(OhlcvModel.timestamp)).where(
            OhlcvModel.symbol == symbol, OhlcvModel.is_ended == True
        )
        result = session.execute(stmt)
        latest_timestamp: datetime | None = result.scalar_one_or_none()

        if latest_timestamp:
            interval_ms = INTERVAL_TO_MS.get(interval, 60000)
            return int(latest_timestamp.timestamp() * 1000) + interval_ms
        else:
            return None


def save_data(OhlcvModel, IndicatorModel, symbol: str, all_klines: list):
    """[수정] 동기식으로 OHLCV와 보조지표를 DB에 저장합니다."""
    if not all_klines:
        return 0
    df = pd.DataFrame(all_klines)
    df["timestamp"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    ohlcv_data_to_save = df[
        ["symbol", "timestamp", "open", "high", "low", "close", "volume", "is_ended"]
    ].to_dict("records")

    df = df.set_index("timestamp").sort_index()
    df.ta.rsi(length=14, append=True, col_names=("rsi_14",))
    df.ta.ema(length=7, append=True, col_names=("ema_7",))
    df.ta.ema(length=21, append=True, col_names=("ema_21",))
    df.ta.ema(length=99, append=True, col_names=("ema_99",))
    df.ta.sma(length=7, append=True, col_names=("sma_7",))
    df.ta.sma(length=21, append=True, col_names=("sma_21",))
    df.ta.sma(length=99, append=True, col_names=("sma_99",))
    df.ta.macd(append=True, col_names=("macd", "macd_hist", "macd_signal"))
    df.ta.bbands(
        length=20,
        append=True,
        col_names=("bb_lower", "bb_middle", "bb_upper", "bb_bandwidth", "bb_percent"),
    )
    df["volume_20"] = df["volume"].rolling(20).mean()
    df = df.dropna()

    indicators_data_to_save = []
    if not df.empty:
        df_reset = df.reset_index()
        indicator_schema_keys = [
            "timestamp",
            "rsi_14",
            "ema_7",
            "ema_21",
            "ema_99",
            "sma_7",
            "sma_21",
            "sma_99",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "volume_20",
        ]
        final_columns = [
            col for col in indicator_schema_keys if col in df_reset.columns
        ]
        indicators_data_to_save = df_reset[final_columns].to_dict("records")
        for row in indicators_data_to_save:
            row["symbol"] = symbol

    with SyncSessionLocal() as session:
        with session.begin():
            if ohlcv_data_to_save:
                ohlcv_stmt = insert(OhlcvModel).values(ohlcv_data_to_save)
                ohlcv_keys = ohlcv_data_to_save[0].keys()
                update_ohlcv_cols = {
                    key: getattr(ohlcv_stmt.excluded, key)
                    for key in ohlcv_keys
                    if key not in ["symbol", "timestamp"]
                }
                ohlcv_stmt = ohlcv_stmt.on_conflict_do_update(
                    index_elements=["symbol", "timestamp"],
                    set_=update_ohlcv_cols,
                )
                session.execute(ohlcv_stmt)

            if indicators_data_to_save:
                ind_stmt = insert(IndicatorModel).values(indicators_data_to_save)
                ind_keys = indicators_data_to_save[0].keys()
                update_ind_cols = {
                    key: getattr(ind_stmt.excluded, key)
                    for key in ind_keys
                    if key not in ["symbol", "timestamp"]
                }
                ind_stmt = ind_stmt.on_conflict_do_update(
                    index_elements=["symbol", "timestamp"],
                    set_=update_ind_cols,
                )
                session.execute(ind_stmt)
        session.commit()
    return len(ohlcv_data_to_save)


# 메모리 배치 처리가 추가된 동기식 Celery 태스크
@celery_app.task(bind=True, name="ohlcv.backfill_symbol_interval")
def backfill_symbol_interval(self: Task, symbol: str, pair: str, interval: str):
    """
    [수정] 동기식으로 실제 데이터 수집 및 진행률 업데이트 로직 (메모리 배치 처리 추가)
    """
    OhlcvModel = OHLCV_MODELS.get(interval)
    IndicatorModel = INDICATOR_MODELS.get(interval)
    if not OhlcvModel or not IndicatorModel:
        raise ValueError(f"지원하지 않는 인터벌입니다: {interval}")

    BATCH_SAVE_SIZE = 1000000
    all_klines_data = []  # 현재 배치를 위한 리스트
    total_saved_count = 0  # 총 저장 개수

    try:
        db_start_time_ms = get_start_time_ms(symbol, interval, OhlcvModel)

        with httpx.Client() as client:
            try:
                server_time_res = client.get("https://fapi.binance.com/fapi/v1/time")
                server_time_res.raise_for_status()
                server_time_ms = server_time_res.json()["serverTime"]
            except Exception as e:
                logger.error(f"[{symbol}-{interval}] Binance 서버 시간 조회 실패: {e}")
                raise Exception(f"Binance 서버 시간 조회 실패: {e}")

        progress_end_ms = server_time_ms
        current_start_time_ms = None
        progress_start_ms = None

        if db_start_time_ms:
            logger.info(f"[{symbol}-{interval}] 증분 업데이트 시작.")
            current_start_time_ms = db_start_time_ms
            progress_start_ms = db_start_time_ms
        else:
            logger.info(
                f"[{symbol}-{interval}] DB가 비어있음. API에서 실제 첫 캔들 시간 조회..."
            )
            with httpx.Client() as client:
                try:
                    params = {
                        "symbol": pair,
                        "interval": interval,
                        "startTime": 1,
                        "limit": 1,
                    }
                    res = client.get(BINANCE_FAPI_URL, params=params)
                    res.raise_for_status()
                    first_candle_data = res.json()

                    if not first_candle_data:
                        logger.warning(
                            f"[{symbol}-{interval}] API에 데이터가 전혀 없습니다. 작업 종료."
                        )
                        return {
                            "status": "COMPLETE",
                            "symbol": symbol,
                            "interval": interval,
                            "saved_count": 0,
                        }

                    actual_first_candle_ms = int(first_candle_data[0][0])
                    current_start_time_ms = actual_first_candle_ms
                    progress_start_ms = actual_first_candle_ms
                    first_candle_dt = datetime.fromtimestamp(
                        actual_first_candle_ms / 1000, tz=timezone.utc
                    )
                    logger.info(
                        f"[{symbol}-{interval}] 실제 시작 시간 확인: {first_candle_dt.isoformat()}"
                    )
                except Exception as e:
                    logger.error(f"[{symbol}-{interval}] 첫 캔들 조회 실패: {e}")
                    raise Exception(f"첫 캔들 조회 실패: {e}")

        if current_start_time_ms is None or progress_start_ms is None:
            raise Exception("Start time could not be determined.")

        last_known_pct = 0.0

        with httpx.Client(timeout=30.0) as client:
            while True:
                params = {
                    "symbol": pair,
                    "interval": interval,
                    "limit": KLINE_LIMIT,
                    "startTime": current_start_time_ms,
                }

                try:
                    res = client.get(BINANCE_FAPI_URL, params=params)

                    if res.status_code == 429 or res.status_code == 418:
                        retry_after = res.headers.get("Retry-After")
                        sleep_time = 60
                        if retry_after and retry_after.isdigit():
                            sleep_time = int(retry_after)

                        logger.warning(
                            f"[{symbol}-{interval}] Rate limit hit (Status {res.status_code}). Sleeping for {sleep_time} seconds..."
                        )
                        self.update_state(
                            state="PROGRESS",
                            meta={
                                "symbol": symbol,
                                "interval": interval,
                                "pct": last_known_pct,
                                "last_candle_time": datetime.fromtimestamp(
                                    current_start_time_ms / 1000, tz=timezone.utc
                                ).isoformat(),
                                "status": f"Rate limit. Paused for {sleep_time}s.",
                            },
                        )
                        time.sleep(sleep_time)
                        continue

                    res.raise_for_status()
                    klines = res.json()

                except httpx.HTTPStatusError as e:
                    logger.error(f"[{symbol}-{interval}] HTTP Error: {e}")
                    raise Exception(f"HTTP Error: {e.response.status_code}")
                except httpx.RequestError as e:
                    logger.error(f"[{symbol}-{interval}] Connection Error: {e}")
                    raise Exception(f"Connection Error: {e}")

                if not klines:
                    logger.info(f"[{symbol}-{interval}] API가 빈 목록 반환. 루프 종료.")
                    break

                last_candle_time_ms = int(klines[-1][0])
                new_klines_count = 0
                for k in klines:
                    open_time_ms = int(k[0])
                    if db_start_time_ms and open_time_ms < db_start_time_ms:
                        continue

                    all_klines_data.append(
                        {
                            "symbol": symbol,
                            "open_time_ms": int(k[0]),
                            "open": float(k[1]),
                            "high": float(k[2]),
                            "low": float(k[3]),
                            "close": float(k[4]),
                            "volume": float(k[5]),
                            "is_ended": True,
                        }
                    )
                    new_klines_count += 1

                # 1. 진행률을 먼저 계산하고 업데이트
                pct = 0
                if progress_start_ms and progress_end_ms > progress_start_ms:
                    pct = (
                        (last_candle_time_ms - progress_start_ms)
                        / (progress_end_ms - progress_start_ms)
                    ) * 100
                last_known_pct = min(round(pct, 2), 100.0)

                self.update_state(
                    state="PROGRESS",
                    meta={
                        "symbol": symbol,
                        "interval": interval,
                        "pct": last_known_pct,
                        "last_candle_time": datetime.fromtimestamp(
                            last_candle_time_ms / 1000, tz=timezone.utc
                        ).isoformat(),
                        "status": "Running...",
                    },
                )

                # 2. 그 다음에 느린 배치 저장을 실행
                if len(all_klines_data) >= BATCH_SAVE_SIZE:
                    logger.info(
                        f"[{symbol}-{interval}] 메모리 배치 {len(all_klines_data)}개 저장 시도..."
                    )
                    saved_in_batch = save_data(
                        OhlcvModel, IndicatorModel, symbol, all_klines_data
                    )
                    total_saved_count += saved_in_batch
                    all_klines_data.clear()  # 메모리 비우기

                if new_klines_count == 0 and len(klines) > 0:
                    logger.info(
                        f"[{symbol}-{interval}] 중복 데이터만 수신됨. 루프 종료."
                    )
                    break

                if len(klines) < KLINE_LIMIT:
                    logger.info(
                        f"[{symbol}-{interval}] 캔들이 {KLINE_LIMIT}개 미만({len(klines)}개)이므로 종료."
                    )
                    break

                current_start_time_ms = last_candle_time_ms + 1

                if current_start_time_ms >= server_time_ms:
                    logger.info(
                        f"[{symbol}-{interval}] 현재 시간까지 모두 수집하여 종료."
                    )
                    break

                try:
                    used_weight = int(res.headers.get(RATE_LIMIT_HEADER, "0"))
                    if used_weight > TARGET_WEIGHT:
                        sleep_duration = 10
                        logger.warning(
                            f"[{symbol}-{interval}] High weight ({used_weight}). Pausing for {sleep_duration}s."
                        )
                        # 여기서도 상태 업데이트를 먼저 수행
                        self.update_state(
                            state="PROGRESS",
                            meta={
                                "symbol": symbol,
                                "interval": interval,
                                "pct": last_known_pct,
                                "last_candle_time": datetime.fromtimestamp(
                                    last_candle_time_ms / 1000, tz=timezone.utc
                                ).isoformat(),
                                "status": f"Pacing weight. Paused for {sleep_duration}s.",
                            },
                        )
                        time.sleep(sleep_duration)
                except Exception:
                    time.sleep(0.5)

        # 루프가 끝난 후, 메모리에 남아있는 나머지 캔들 저장
        if all_klines_data:
            logger.info(
                f"[{symbol}-{interval}] 마지막 남은 배치 {len(all_klines_data)}개 캔들 저장 시도..."
            )
            saved_in_batch = save_data(
                OhlcvModel, IndicatorModel, symbol, all_klines_data
            )
            total_saved_count += saved_in_batch
            all_klines_data.clear()

        return {
            "status": "COMPLETE",
            "symbol": symbol,
            "interval": interval,
            "saved_count": total_saved_count,
        }
    except Exception as e:
        logger.error(
            f"Task {self.request.id} (Symbol: {symbol}, Interval: {interval}) failed: {e}"
        )
        raise Exception(f"Task failed for {symbol} {interval}: {str(e)}")
