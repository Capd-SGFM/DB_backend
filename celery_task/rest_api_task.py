# ================================================================
#  완전 패치된 BACKFILL 엔진 (무한 대기 0%, 즉시 SUCCESS 반환)
# ================================================================
import time
from datetime import datetime, timezone
from typing import Tuple, Optional

import httpx
import pandas as pd
from celery import Task
from loguru import logger
from sqlalchemy import select, func, text
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

    with SyncSessionLocal() as session, session.begin():
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
#  DB 시작 시각 계산
# ================================================================
def get_start_time_ms(
    symbol, interval, OhlcvModel, ws_frontier_ms
) -> Tuple[Optional[int], bool]:
    ws_frontier_dt = datetime.fromtimestamp(ws_frontier_ms / 1000, tz=timezone.utc)

    with SyncSessionLocal() as session:
        total_count = session.execute(
            select(func.count()).where(OhlcvModel.symbol == symbol)
        ).scalar_one()

        has_any_row = total_count > 0

        stmt = select(func.max(OhlcvModel.timestamp)).where(
            OhlcvModel.symbol == symbol,
            OhlcvModel.is_ended == True,
            OhlcvModel.timestamp < ws_frontier_dt,
        )
        latest_ts = session.execute(stmt).scalar_one_or_none()

    if latest_ts:
        interval_ms = INTERVAL_TO_MS.get(interval, 60_000)
        return int(latest_ts.timestamp() * 1000) + interval_ms, has_any_row

    return None, has_any_row


# ================================================================
#  OHLCV 저장
# ================================================================
def save_data(OhlcvModel, symbol, rows):
    if not rows:
        return 0

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    recs = df[
        ["symbol", "timestamp", "open", "high", "low", "close", "volume", "is_ended"]
    ].to_dict("records")

    with SyncSessionLocal() as session:
        stmt = insert(OhlcvModel).values(recs)
        update_cols = {
            key: getattr(stmt.excluded, key)
            for key in recs[0].keys()
            if key not in ["symbol", "timestamp"]
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "timestamp"], set_=update_cols
        )
        session.execute(stmt)

    return len(recs)


# ================================================================
#  REST 백필 태스크 (완전 패치)
# ================================================================
@celery_app.task(bind=True, name="ohlcv.backfill_symbol_interval")
def backfill_symbol_interval(
    self: Task, symbol, pair, interval, ws_frontier_ms=None, run_id=None
):

    try:
        OhlcvModel = OHLCV_MODELS.get(interval)
        if not OhlcvModel:
            raise ValueError(f"지원하지 않는 interval: {interval}")

        # pipeline OFF면 즉시 종료
        if not is_pipeline_active():
            upsert_backfill_progress(run_id, symbol, interval, "PENDING", 0, None, None)
            return {"status": "SKIP", "symbol": symbol, "interval": interval}

        # ws_frontier 없으면 조회
        if ws_frontier_ms is None:
            with httpx.Client(timeout=20) as client:
                r = client.get("https://fapi.binance.com/fapi/v1/time")
                ws_frontier_ms = int(r.json()["serverTime"])

        # 시작점 계산
        db_start_ms, has_any_row = get_start_time_ms(
            symbol, interval, OhlcvModel, ws_frontier_ms
        )

        # ---- 이미 최신이면 즉시 SUCCESS ----
        if has_any_row and db_start_ms and db_start_ms >= ws_frontier_ms:
            upsert_backfill_progress(
                run_id, symbol, interval, "SUCCESS", 100, None, None
            )
            return {
                "status": "COMPLETE",
                "symbol": symbol,
                "interval": interval,
                "saved": 0,
            }

        # ---- DB에 데이터가 없으면 Binance 첫 캔들 시각 찾기 ----
        if not has_any_row or db_start_ms is None:
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
                    # Binance 데이터조차 없음 → SUCCESS 처리
                    upsert_backfill_progress(
                        run_id, symbol, interval, "SUCCESS", 100, None, None
                    )
                    return {
                        "status": "COMPLETE",
                        "symbol": symbol,
                        "interval": interval,
                    }

                first_ms = int(arr[0][0])
                start_ms = first_ms
        else:
            start_ms = db_start_ms

        # ---- 시작점이 frontier 이후 → SUCCESS ----
        if start_ms >= ws_frontier_ms:
            upsert_backfill_progress(
                run_id, symbol, interval, "SUCCESS", 100, None, None
            )
            return {"status": "COMPLETE", "symbol": symbol, "interval": interval}

        # =======================================================
        #  메인 수집 루프
        # =======================================================
        interval_ms = INTERVAL_TO_MS.get(interval, 60_000)
        buffer = []
        total_saved = 0
        progress_start = start_ms
        progress_end = ws_frontier_ms

        with httpx.Client(timeout=20) as client:
            while True:
                if not is_pipeline_active():
                    upsert_backfill_progress(
                        run_id, symbol, interval, "PENDING", 0, None, None
                    )
                    return {"status": "SKIP"}

                if start_ms >= ws_frontier_ms:
                    break

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
                r.raise_for_status()
                arr = r.json()

                # ----- API 결과가 아예 없으면 SUCCESS -----
                if not arr:
                    upsert_backfill_progress(
                        run_id, symbol, interval, "SUCCESS", 100, None, None
                    )
                    return {
                        "status": "COMPLETE",
                        "symbol": symbol,
                        "interval": interval,
                    }

                new_count = 0
                last_open = None

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
                    last_open = open_ms
                    new_count += 1

                # ---- 새로 저장된 것이 0개 → SUCCESS ----
                if new_count == 0:
                    upsert_backfill_progress(
                        run_id, symbol, interval, "SUCCESS", 100, None, None
                    )
                    return {
                        "status": "COMPLETE",
                        "symbol": symbol,
                        "interval": interval,
                    }

                # ---- 진행률 ----
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

                # ---- 배치 저장 ----
                if len(buffer) >= 50_000:
                    total_saved += save_data(OhlcvModel, symbol, buffer)
                    buffer.clear()

                start_ms = last_open + interval_ms

            # ------ 마지막 배치 저장 ------
            if buffer:
                total_saved += save_data(OhlcvModel, symbol, buffer)

        # 최종 SUCCESS
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
