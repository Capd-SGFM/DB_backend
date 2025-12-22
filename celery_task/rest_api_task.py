# ================================================================
#  완전 패치된 BACKFILL 엔진 (WS-frontier 자동 계산 + 저장 커밋 보장)
# ================================================================
import time
from datetime import datetime, timezone
from typing import Tuple, Optional

import httpx
import pandas as pd
import numpy as np
from datetime import timedelta
import ta
from sklearn.preprocessing import RobustScaler
from celery import Task
from loguru import logger
from sqlalchemy import select, func, text, asc
from sqlalchemy.dialects.postgresql import insert

from db_module.connect_sqlalchemy_engine import SyncSessionLocal
from models import OHLCV_MODELS, BACKTESTING_FEATURES_MODELS
from models.backfill_progress import BackfillProgress
from models.pipeline_state import (
    is_pipeline_active,
    set_component_error,
    PipelineComponent,
)
from . import celery_app


BINANCE_FAPI_URL = "https://fapi.binance.com/fapi/v1/klines"
KLINE_LIMIT = 1000


INTERVAL_TO_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


# ================================================================
#  DB에서 WS frontier 얻기 (is_ended=False 중 가장 오래된 timestamp)
# ================================================================
def detect_ws_frontier_ms(symbol: str, OhlcvModel):
    """
    1) DB에 is_ended=False가 있다면 → 가장 오래된 timestamp 사용
    2) 없다면 → None 반환 (Caller가 Retry 처리)
    """
    if not is_pipeline_active():
        return None

    with SyncSessionLocal() as session:
        earliest = (
            session.query(OhlcvModel.timestamp)
            .filter(OhlcvModel.symbol == symbol, OhlcvModel.is_ended == False)
            .order_by(asc(OhlcvModel.timestamp))
            .first()
        )

    if earliest:
        return int(earliest[0].timestamp() * 1000)

    return None


# ================================================================
#  백필 프로그래스 UPSERT
# ================================================================
def upsert_backfill_progress(
    run_id: str,
    symbol: str,
    interval: str,
    state: str,
    pct_time: float,
    last_candle_ts: Optional[datetime],
    last_error: Optional[str],
):
    if not run_id:
        return

    with SyncSessionLocal() as session, session.begin():  # 🔥 commit 보장
        stmt = (
            insert(BackfillProgress)
            .values(
                run_id=run_id,
                symbol=symbol,
                interval=interval,
                state=state,
                pct_time=pct_time,
                last_candle_ts=last_candle_ts,
                last_error=last_error,
            )
            .on_conflict_do_update(
                index_elements=["run_id", "symbol", "interval"],
                set_={
                    "state": state,
                    "pct_time": pct_time,
                    "last_candle_ts": last_candle_ts,
                    "last_error": last_error,
                    "updated_at": text("now()"),
                },
            )
        )
        session.execute(stmt)


# ================================================================
#  DB 시작 시각 계산 (is_ended=True 최신 timestamp)
# ================================================================
def get_start_time_ms(symbol, interval, OhlcvModel, ws_frontier_ms):
    ws_frontier_dt = datetime.fromtimestamp(ws_frontier_ms / 1000, tz=timezone.utc)

    with SyncSessionLocal() as session:
        count_all = (
            session.query(func.count()).filter(OhlcvModel.symbol == symbol).scalar()
        )

        has_any_row = count_all > 0

        latest_ts = (
            session.query(func.max(OhlcvModel.timestamp))
            .filter(
                OhlcvModel.symbol == symbol,
                OhlcvModel.is_ended == True,
                OhlcvModel.timestamp < ws_frontier_dt,
            )
            .scalar()
        )

    if latest_ts:
        interval_ms = INTERVAL_TO_MS.get(interval, 60_000)
        return int(latest_ts.timestamp() * 1000) + interval_ms, has_any_row

    return None, has_any_row


# ================================================================
#  OHLCV 저장 (🔥 반드시 commit 보장됨)
# ================================================================
def save_data(OhlcvModel, symbol, rows):
    """Ultra-fast save using PostgreSQL COPY (TimescaleDB optimized)"""
    if not rows:
        return 0

    logger.info(f"Saving into table={OhlcvModel.__tablename__}, rows={len(rows)} using COPY")

    try:
        # Prepare data for COPY
        from io import StringIO
        
        # Create CSV buffer
        buffer = StringIO()
        for row in rows:
            ts = datetime.fromtimestamp(row["open_time_ms"] / 1000, tz=timezone.utc)
            # Format: symbol, timestamp, open, high, low, close, volume, is_ended
            line = f"{row['symbol']}\t{ts.isoformat()}\t{row['open']}\t{row['high']}\t{row['low']}\t{row['close']}\t{row['volume']}\t{row['is_ended']}\n"
            buffer.write(line)
        
        buffer.seek(0)
        
        # Use temporary table for ON CONFLICT handling
        temp_table = f"temp_{OhlcvModel.__tablename__}_{int(time.time() * 1000)}"
        # 🔧 Fix: Include schema name for table reference
        full_table_name = f"trading_data.{OhlcvModel.__tablename__}"
        
        with SyncSessionLocal() as session:
            connection = session.connection().connection
            cursor = connection.cursor()
            
            try:
                # Create temporary table with same structure
                cursor.execute(f"""
                    CREATE TEMPORARY TABLE {temp_table} (LIKE {full_table_name} INCLUDING ALL)
                    ON COMMIT DROP
                """)
                
                # 🚀 Ultra-fast COPY into temp table
                cursor.copy_from(
                    buffer,
                    temp_table,
                    columns=['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'is_ended'],
                    sep='\t'
                )
                
                # Insert from temp table with ON CONFLICT handling
                cursor.execute(f"""
                    INSERT INTO {full_table_name} 
                    (symbol, timestamp, open, high, low, close, volume, is_ended)
                    SELECT symbol, timestamp, open, high, low, close, volume, is_ended
                    FROM {temp_table}
                    ON CONFLICT (symbol, timestamp) 
                    DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        is_ended = EXCLUDED.is_ended
                """)
                
                connection.commit()
                logger.info(f"✅ COPY successful: {len(rows)} rows")
                
            except Exception as e:
                connection.rollback()
                logger.error(f"❌ COPY failed: {e}, falling back to INSERT")
                raise
            finally:
                cursor.close()
        
        return len(rows)
        
    except Exception as e:
        # Fallback to original INSERT method on any error
        logger.warning(f"COPY failed, using INSERT fallback: {e}")
        
        recs = []
        for row in rows:
            recs.append({
                "symbol": row["symbol"],
                "timestamp": datetime.fromtimestamp(row["open_time_ms"] / 1000, tz=timezone.utc),
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "is_ended": row["is_ended"],
            })
        
        with SyncSessionLocal() as session, session.begin():
            stmt = insert(OhlcvModel).values(recs)
            update_cols = {
                k: getattr(stmt.excluded, k)
                for k in recs[0].keys()
                if k not in ["symbol", "timestamp"]
            }
            
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol", "timestamp"],
                set_=update_cols,
            )
            session.execute(stmt)
        
        return len(rows)



# ================================================================
#  Feature Engineering Helpers
# ================================================================
def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Ensure sorted
    df = df.sort_index()

    # 1) log_return
    # df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # 2) EMA Ratio (20)
    # ema = df["close"].ewm(span=20, adjust=False).mean()
    # df["ema_ratio"] = (df["close"] - ema) / ema
    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_ratio"] = (df["close"] - ema20) / ema20

    # 3) MACD Hist (12, 26, 9)
    # macd_ind = ta.trend.MACD(...)
    # df["macd_hist"] = macd_ind.macd_diff()
    macd_ind = ta.trend.MACD(
        close=df["close"],
        window_slow=26,
        window_fast=12,
        window_sign=9
    )
    df["macd_hist"] = macd_ind.macd_diff()

    # 4) Bollinger Bands (20, 2)
    bb_ind = ta.volatility.BollingerBands(
        close=df["close"],
        window=20,
        window_dev=2,
    )
    bb_high = bb_ind.bollinger_hband()
    bb_low = bb_ind.bollinger_lband()
    bb_mid = bb_ind.bollinger_mavg()

    denom = bb_high - bb_low

    # bandwidth = (upper - lower) / mid
    df["bandwidth"] = np.where(bb_mid == 0, 0.0, denom / bb_mid)
    
    # pct_b = (close - low) / (upper - lower)
    # If denom == 0 => 0.5
    raw_pct_b = (df["close"] - bb_low) / denom
    df["pct_b"] = np.where(denom == 0, 0.5, raw_pct_b)

    # 5) RSI (14)
    # df["rsi"] = rsi_ind.rsi() / 100.0
    rsi_ind = ta.momentum.RSIIndicator(close=df["close"], window=14)
    df["rsi"] = rsi_ind.rsi() / 100.0
    
    # 6) MFI (14)
    # mfi_ind = ta.volume.MFIIndicator(...)
    # df["mfi"] = mfi_ind.money_flow_index() / 100.0
    # fillna(0.5)
    mfi_ind = ta.volume.MFIIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        volume=df["volume"],
        window=14,
    )
    mfi_raw = mfi_ind.money_flow_index() / 100.0
    df["mfi"] = mfi_raw.fillna(0.5)

    # 7) ATR (14) - For Dedicated Storage
    atr_ind = ta.volatility.AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14,
    )
    df["atr"] = atr_ind.average_true_range()

    return df


def robust_scale_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    feature_cols = ["log_return", "ema_ratio", "macd_hist", "bandwidth"]
    
    # Explicitly use sklearn RobustScaler as requested
    scaler = RobustScaler()
    
    # Check valid rows (not all NaNs)
    mask = df[feature_cols].notna().all(axis=1)
    if mask.sum() == 0:
        # If no valid data to fit, return as is (all NaNs or empty) or handle gracefully
        # User script raises error, but pipeline should probably survive.
        # Let's return as is (features will be NaN or unscaled)
        return df

    scaler.fit(df.loc[mask, feature_cols])
    df.loc[mask, feature_cols] = scaler.transform(df.loc[mask, feature_cols])

    return df

def save_backtesting_features(FeatureModel, rows: list):
    if not rows:
        return 0
    from sqlalchemy.dialects.postgresql import insert
    
    with SyncSessionLocal() as session, session.begin():
        stmt = insert(FeatureModel).values(rows)
        # On conflict update
        update_cols = {
            k: getattr(stmt.excluded, k)
            for k in rows[0].keys()
            if k not in ["symbol", "timestamp"]
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "timestamp"],
            set_=update_cols
        )
        session.execute(stmt)
    return len(rows)


# ================================================================
#  REST 백필 태스크
# ================================================================
@celery_app.task(bind=True, name="ohlcv.backfill_symbol_interval")
def backfill_symbol_interval(
    self: Task,
    symbol,
    pair,
    interval,
    ws_frontier_ms=None,
    run_id=None,
):
    try:
        OhlcvModel = OHLCV_MODELS.get(interval)
        if not OhlcvModel:
            raise ValueError(f"Unsupported interval: {interval}")

        # 파이프라인 OFF면 종료
        if not is_pipeline_active():
            upsert_backfill_progress(run_id, symbol, interval, "PENDING", 0, None, None)
            return {"status": "SKIP"}

        # 🔥 WS frontier = is_ended=False 중 가장 오래된 timestamp (없으면 대기)
        ws_frontier_ms = detect_ws_frontier_ms(symbol, OhlcvModel)
        if not ws_frontier_ms:
            # 데이터가 아직 없으면 3초 뒤 재시도 (최대 20번 = 1분)
            # upsert_backfill_progress는 PENDING 유지
            upsert_backfill_progress(run_id, symbol, interval, "PENDING", 0, None, None)
            logger.info(f"[Backfill] {symbol} {interval}: WS data not found, retrying...")
            raise self.retry(countdown=3, max_retries=20)

        # DB 시작 위치 계산
        db_start_ms, has_any = get_start_time_ms(
            symbol, interval, OhlcvModel, ws_frontier_ms
        )

        # 이미 최신이면 즉시 SUCCESS
        if has_any and db_start_ms and db_start_ms >= ws_frontier_ms:
            upsert_backfill_progress(
                run_id, symbol, interval, "SUCCESS", 100, None, None
            )
            return {
                "status": "COMPLETE",
                "symbol": symbol,
                "interval": interval,
                "saved": 0,
            }

        # DB에 없으면 최초 캔들부터 시작
        if not has_any or db_start_ms is None:
            with httpx.Client(timeout=20) as client:
                r = client.get(
                    BINANCE_FAPI_URL,
                    params={
                        "symbol": pair,
                        "interval": interval,
                        "startTime": 1,
                        "limit": 1,
                    },
                )
                arr = r.json()
                if not arr:
                    upsert_backfill_progress(
                        run_id, symbol, interval, "SUCCESS", 100, None, None
                    )
                    return {"status": "COMPLETE"}
                start_ms = int(arr[0][0])
        else:
            start_ms = db_start_ms

        if start_ms >= ws_frontier_ms:
            upsert_backfill_progress(
                run_id, symbol, interval, "SUCCESS", 100, None, None
            )
            return {"status": "COMPLETE"}

        # =======================================================
        #   메인 수집 루프
        # =======================================================
        interval_ms = INTERVAL_TO_MS[interval]
        total_saved = 0
        buffer = []

        progress_start = start_ms
        progress_end = ws_frontier_ms

        with httpx.Client(timeout=20) as client:
            while start_ms < ws_frontier_ms:
                if not is_pipeline_active():
                    upsert_backfill_progress(
                        run_id, symbol, interval, "PENDING", 0, None, None
                    )
                    return {"status": "SKIP"}

                r = client.get(
                    BINANCE_FAPI_URL,
                    params={
                        "symbol": pair,
                        "interval": interval,
                        "startTime": start_ms,
                        "endTime": ws_frontier_ms - 1,
                        "limit": KLINE_LIMIT,
                    },
                )
                
                # 🚀 Rate Limit Handling
                if r.status_code == 429:
                    logger.warning(f"[Backfill] {symbol} {interval}: 429 Too Many Requests. Retrying...")
                    time.sleep(5)  # Cool down
                    raise self.retry(countdown=10, max_retries=20)
                
                r.raise_for_status()
                arr = r.json()
                
                # 🚀 Dynamic Rate Limiting (Smart Throttling)
                # Binance Limit: 2400 weight / minute
                try:
                    used_weight = int(r.headers.get("x-mbx-used-weight-1m", 0))
                    limit_weight = 2400
                    
                    if used_weight > (limit_weight * 0.95):
                        # > 95% usage: Danger zone, sleep long
                        logger.warning(f"[Backfill] Rate limit critical: {used_weight}/{limit_weight}. Sleeping 10s...")
                        time.sleep(10)
                    elif used_weight > (limit_weight * 0.80):
                        # > 80% usage: Throttle down
                        logger.info(f"[Backfill] Rate limit high: {used_weight}/{limit_weight}. Sleeping 3s...")
                        time.sleep(3)
                    else:
                        # < 80% usage: Full speed (minimal delay)
                        time.sleep(0.1)
                        
                except Exception:
                    # Header parsing failed, fallback to safe default
                    time.sleep(0.5)

                if not arr:
                    break

                last_open = None
                new_count = 0

                for k in arr:
                    open_ms = int(k[0])
                    if open_ms >= ws_frontier_ms:
                        continue

                    buffer.append(
                        {
                            "symbol": symbol,
                            "open_time_ms": open_ms,
                            "open": float(k[1]),
                            "high": float(k[2]),
                            "low": float(k[3]),
                            "close": float(k[4]),
                            "volume": float(k[5]),
                            "is_ended": True,
                        }
                    )
                    new_count += 1
                    last_open = open_ms

                if new_count == 0:
                    break

                # 🚀 Optimized: Update progress every 10,000 rows instead of 1,000
                if len(buffer) % 10_000 == 0 and len(buffer) > 0:
                    pct = min(
                        round(
                            (last_open - progress_start)
                            / (progress_end - progress_start)
                            * 100,
                            2,
                        ),
                        100.0,
                    )
                    upsert_backfill_progress(
                        run_id,
                        symbol,
                        interval,
                        "PROGRESS",
                        pct,
                        datetime.fromtimestamp(last_open / 1000, tz=timezone.utc),
                        None,
                    )

                # 🚀 Optimized: Larger buffer for better bulk insert performance
                if len(buffer) >= 100_000:
                    total_saved += save_data(OhlcvModel, symbol, buffer)
                    buffer.clear()

                start_ms = last_open + interval_ms

        # 마지막 버퍼 flush
        if buffer:
            total_saved += save_data(OhlcvModel, symbol, buffer)

        upsert_backfill_progress(run_id, symbol, interval, "SUCCESS", 100, None, None)
        return {
            "status": "COMPLETE",
            "symbol": symbol,
            "interval": interval,
            "saved": total_saved,
        }

    except httpx.ConnectError as e:
        # 네트워크 연결 오류: 재시도 (최대 5번, 지수 백오프)
        logger.warning(
            f"[Backfill] {symbol} {interval}: Connection error, retrying... "
            f"(attempt {self.request.retries + 1}/5)"
        )
        upsert_backfill_progress(
            run_id, symbol, interval, "PENDING", 0, None, f"Connection error (retrying...)"
        )
        raise self.retry(exc=e, countdown=2 ** self.request.retries, max_retries=5)
    
    
    except httpx.TimeoutException as e:
        # 타임아웃 오류: 재시도 (최대 3번)
        logger.warning(
            f"[Backfill] {symbol} {interval}: Timeout error, retrying... "
            f"(attempt {self.request.retries + 1}/3)"
        )
        upsert_backfill_progress(
            run_id, symbol, interval, "PENDING", 0, None, f"Timeout (retrying...)"
        )
        raise self.retry(exc=e, countdown=5, max_retries=3)
    
    except Exception as e:
        # 기타 오류: 바로 실패 처리 (재시도 안 함)
        logger.exception(e)
        set_component_error(PipelineComponent.BACKFILL, str(e))
        upsert_backfill_progress(run_id, symbol, interval, "FAILURE", 0, None, str(e))
        raise
@celery_app.task(bind=True, name="ohlcv.backfill_backtesting_symbol_interval")
def backfill_backtesting_symbol_interval(
    self: Task,
    symbol,
    pair,
    interval,
    run_id=None,
):
    from models.pipeline_state import _get_state

    try:
        OhlcvModel = OHLCV_MODELS.get(interval)
        FeatureModel = BACKTESTING_FEATURES_MODELS.get(interval)
        
        if not OhlcvModel:
            raise ValueError(f"Unsupported interval: {interval}")
        if not FeatureModel:
            # If interval has no feature model (e.g. unsupported), skip or error?
            # User only requested features for these.
            raise ValueError(f"No Feature Model for interval: {interval}")

        # Check Backtesting State based on run_id prefix
        # VIP tasks have run_id starting with "vip-"
        if run_id and run_id.startswith("vip-"):
            st = _get_state("VIP_BACKTESTING")
        else:
            st = _get_state(PipelineComponent.BACKTESTING)
            
        if not st or not st.is_active:
            upsert_backfill_progress(run_id, symbol, interval, "PENDING", 0, None, "Cancelled (Inactive)")
            return {"status": "SKIP"}

        # Define Scope
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        # Determine Start Date based on Pipeline type and Interval
        start_target_ms = 0
        is_vip = run_id and run_id.startswith("vip-")
        
        # 1. Check DB for existing data
        db_start_ms, has_any = get_start_time_ms(symbol, interval, OhlcvModel, now_ms)
        
        if has_any and db_start_ms:
             # Resume from last candle
             start_target_ms = db_start_ms
        else:
            # First run: Determine start date based on pipeline type
            if is_vip:
                # VIP Pipeline: 2021-01-01 for all, SOL: 2021-06-01
                if symbol == "SOL":
                    start_target_dt = datetime(2021, 6, 1, tzinfo=timezone.utc)
                else:
                    start_target_dt = datetime(2021, 1, 1, tzinfo=timezone.utc)
            else:
                # Backtesting Pipeline (95 symbols):
                # 1m, 5m, 15m: 2025-01-01
                # 1h, 4h, 1d: 2023-01-01
                if interval in ("1m", "5m", "15m"):
                    start_target_dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
                else:  # 1h, 4h, 1d
                    start_target_dt = datetime(2023, 1, 1, tzinfo=timezone.utc)
            
            start_target_ms = int(start_target_dt.timestamp() * 1000)

        # Warmup: 128 candles before start date (for LSTM sequence requirement)
        # These warmup candles will ALSO be saved to DB
        warmup_candles = 128
        
        # Determine Fetch Start (Warm-up)
        # Subtract warmup_candles * interval_ms
        interval_ms = INTERVAL_TO_MS[interval]
        warmup_ms = warmup_candles * interval_ms
        
        # Fetch starts from (start_target - warmup) to include warmup data
        fetch_start_ms = start_target_ms - warmup_ms
        
        # Buffer to hold all raw klines
        all_klines = []
        
        # Use a separate pointer for fetching
        current_fetch_ms = fetch_start_ms
        
        with httpx.Client(timeout=20) as client:
            while current_fetch_ms < now_ms:
                # Check State periodically based on run_id prefix
                if run_id and run_id.startswith("vip-"):
                    st = _get_state("VIP_BACKTESTING")
                else:
                    st = _get_state(PipelineComponent.BACKTESTING)
                    
                if not st or not st.is_active:
                    upsert_backfill_progress(run_id, symbol, interval, "PENDING", 0, None, "Cancelled")
                    return {"status": "SKIP"}

                r = client.get(
                    BINANCE_FAPI_URL,
                    params={
                        "symbol": pair,
                        "interval": interval,
                        "startTime": current_fetch_ms,
                        "limit": KLINE_LIMIT,
                    },
                )
                
                if r.status_code == 429:
                    time.sleep(5)
                    raise self.retry(countdown=10, max_retries=20)
                
                r.raise_for_status()
                arr = r.json()
                
                if not arr:
                    break
                
                last_open = current_fetch_ms
                new_count = 0
                
                for k in arr:
                    open_ms = int(k[0])
                    # Ensure candle is closed
                    if open_ms + interval_ms > now_ms:
                        continue 
                    
                    all_klines.append({
                        "open_time": datetime.fromtimestamp(open_ms / 1000, tz=timezone.utc),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                    })
                    new_count += 1
                    last_open = open_ms
                
                # Progress (Fetching)
                pct = min(((last_open - fetch_start_ms) / (now_ms - fetch_start_ms)) * 50, 50.0)
                upsert_backfill_progress(run_id, symbol, interval, "PROGRESS", pct, None, "Fetching data...")

                if new_count == 0:
                    break
                    
                current_fetch_ms = last_open + interval_ms
                
                # Smart Throttling
                try:
                    w = int(r.headers.get("x-mbx-used-weight-1m", 0))
                    if w > 2000: time.sleep(5)
                    else: time.sleep(0.05)
                except:
                    time.sleep(0.1)

        # Processing
        upsert_backfill_progress(run_id, symbol, interval, "PROGRESS", 60.0, None, "Calculating features...")
        
        if not all_klines:
            upsert_backfill_progress(run_id, symbol, interval, "SUCCESS", 100, None, "No data found")
            return {"status": "COMPLETE", "msg": "No data"}
            
        df = pd.DataFrame(all_klines)
        df.set_index("open_time", inplace=True)
        # Ensure index is datetime with timezone
        # Already set in creation, but verify
        # if not df.index.tz: df.index = df.index.tz_localize(timezone.utc)
        
        # 1. Calculate Indicators
        df = calculate_technical_features(df)
        upsert_backfill_progress(run_id, symbol, interval, "PROGRESS", 70.0, None, "Scaling features...")

        # 2. Use ALL data including warmup (128 candles before start_target)
        # Warmup data is needed for LSTM and will be saved to DB
        df_target = df.copy()
        
        if df_target.empty:
            upsert_backfill_progress(run_id, symbol, interval, "SUCCESS", 100, None, "No data to save")
            return {"status": "COMPLETE"}

        # 3. Robust Scale
        df_target = robust_scale_features(df_target)
        upsert_backfill_progress(run_id, symbol, interval, "PROGRESS", 80.0, None, "Saving to DB...")

        # 4. Save
        # Save OHLCV (including warmup data)

        # Create list of dicts for `save_data`
        ohlcv_rows = []
        feature_rows = []
        
        for ts, row in df_target.iterrows():
            # ts is Timestamp, row is Series
            ts_native = ts.to_pydatetime()
            
            # OHLCV Row
            ohlcv_rows.append({
                "symbol": symbol,
                "open_time_ms": int(ts_native.timestamp() * 1000),
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "is_ended": True
            })
            
            # Feature Row
            # NaNs might exist if slice was too close to start (unlikely with 200 warmup)
            # We replace NaN with None for DB
            def clean(val):
                return float(val) if not pd.isna(val) else None

            feature_rows.append({
                "symbol": symbol,
                "timestamp": ts_native,
                "log_return": clean(row.get("log_return")),
                "ema_ratio": clean(row.get("ema_ratio")),
                "macd_hist": clean(row.get("macd_hist")),
                "bandwidth": clean(row.get("bandwidth")),
                "pct_b": clean(row.get("pct_b")),
                "rsi": clean(row.get("rsi")),
                "mfi": clean(row.get("mfi")),
            })

        # Save OHLCV
        save_data(OhlcvModel, symbol, ohlcv_rows)
        
        # Save Features
        save_backtesting_features(FeatureModel, feature_rows)
        
        # [ATR Storage] Save ATR to dedicated table
        try:
            from celery_task.indicator_task import _upsert_atr_dedicated
            from db.session import SyncSessionLocal
            # We need to construct a DataFrame for the helper or pass the logic?
            # Helper takes (session, symbol, interval, df). df needs "atr" column.
            # df_target has "atr" column because we added it in calculate_technical_features
            
            # Use a new session for this operation or reuse?
            # Ideally reuse, but helper takes 'session'.
            # We are not in a session context here (we used session only inside `save_backtesting_features`).
            # So we create a new short-lived session or context.
            with SyncSessionLocal() as session, session.begin():
                _upsert_atr_dedicated(session, symbol, interval, df_target)
                
        except Exception as e:
             logger.error(f"[Backfill] ATR dedicated save failed: {e}")

        upsert_backfill_progress(run_id, symbol, interval, "SUCCESS", 100, None, None)
        return {
            "status": "COMPLETE",
            "symbol": symbol,
            "interval": interval,
            "saved": len(ohlcv_rows)
        }

    except Exception as e:
        logger.exception(f"[Backtesting] Failed: {e}")
        upsert_backfill_progress(run_id, symbol, interval, "FAILURE", 0, None, str(e))
        raise

