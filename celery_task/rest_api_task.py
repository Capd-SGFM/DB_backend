# ================================================================
#  완전 패치된 BACKFILL 엔진 (WS-frontier 자동 계산 + 저장 커밋 보장)
# ================================================================
import time
from datetime import datetime, timezone
from typing import Tuple, Optional

import httpx
import pandas as pd
from celery import Task
from loguru import logger
from sqlalchemy import select, func, text, asc
from sqlalchemy.dialects.postgresql import insert

from db_module.connect_sqlalchemy_engine import SyncSessionLocal
from models import OHLCV_MODELS
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
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
    "1w": 7 * 86_400_000,
    "1M": 30 * 86_400_000,
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
    if not rows:
        return 0

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    recs = df[
        ["symbol", "timestamp", "open", "high", "low", "close", "volume", "is_ended"]
    ].to_dict("records")

    logger.info(f"Saving into table={OhlcvModel.__tablename__}, rows={len(recs)}")

    # 🔥 commit 보장
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

    return len(recs)


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
                arr = r.json()
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

                if len(buffer) >= 50_000:
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

    except Exception as e:
        logger.exception(e)
        set_component_error(PipelineComponent.BACKFILL, str(e))
        upsert_backfill_progress(run_id, symbol, interval, "FAILURE", 0, None, str(e))
        raise
