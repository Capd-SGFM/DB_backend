# celery_task/rest_maintenance_task.py

import uuid
import httpx
from celery import shared_task
from loguru import logger

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.sql import text

from db_module.connect_sqlalchemy_engine import SyncSessionLocal
from models import CryptoInfo, OHLCV_MODELS
from models.pipeline_state import (
    is_pipeline_active,
    set_component_error,
    PipelineComponent,
)
from models.rest_progress import RestProgress
from .rest_api_task import backfill_symbol_interval


# =========================================================
#   RestProgress Upsert 함수
# =========================================================
def upsert_rest_progress(run_id, symbol, interval, state, last_ts=None, error=None):
    with SyncSessionLocal() as session, session.begin():
        stmt = (
            insert(RestProgress)
            .values(
                run_id=run_id,
                symbol=symbol,
                interval=interval,
                state=state,
                last_candle_ts=last_ts,
                last_error=error,
            )
            .on_conflict_do_update(
                index_elements=["run_id", "symbol", "interval"],
                set_={
                    "state": state,
                    "last_candle_ts": last_ts,
                    "last_error": error,
                    "updated_at": text("NOW()"),
                },
            )
        )
        session.execute(stmt)


# =========================================================
#   REST 유지보수 메인 태스크 — 전체 코드
# =========================================================
@shared_task(name="pipeline.run_rest_maintenance")
def run_rest_maintenance():
    """REST 유지보수:
    - WebSocket 누락된 부분을 REST로 채움
    - 유지보수할 데이터가 없으면 즉시 SUCCESS
    """
    logger.info("[REST] run_rest_maintenance 시작")

    if not is_pipeline_active():
        logger.info("[REST] pipeline inactive → 종료")
        return {"status": "INACTIVE"}

    # ----------------------------------
    # 1) run_id 생성
    # ----------------------------------
    run_id = f"rest-{uuid.uuid4().hex}"
    logger.info(f"[REST] rest_run_id={run_id}")

    # ----------------------------------
    # 2) serverTime 조회
    # ----------------------------------
    try:
        with httpx.Client(timeout=10.0) as client:
            res = client.get("https://fapi.binance.com/fapi/v1/time")
            res.raise_for_status()
            server_time_ms = int(res.json()["serverTime"])
    except Exception as e:
        logger.error(f"[REST] serverTime 조회 실패: {e}")
        set_component_error(PipelineComponent.REST_MAINTENANCE, str(e))
        return {"status": "FAILURE", "error": str(e)}

    ws_cutoff = server_time_ms

    # ----------------------------------
    # 3) 심볼 가져오기
    # ----------------------------------
    with SyncSessionLocal() as session:
        symbols = (
            session.query(CryptoInfo.symbol, CryptoInfo.pair)
            .filter(CryptoInfo.pair.isnot(None))
            .all()
        )

    intervals = list(OHLCV_MODELS.keys())

    # ----------------------------------
    # 4) 작업 수행
    # ----------------------------------
    total_jobs = 0
    success_cnt = 0

    for sym, pair in symbols:
        for interval in intervals:
            total_jobs += 1

            # (1) 작업 시작 → PROGRESS
            upsert_rest_progress(run_id, sym, interval, "PROGRESS")

            try:
                result = backfill_symbol_interval(
                    symbol=sym,
                    pair=pair,
                    interval=interval,
                    ws_frontier_ms=ws_cutoff,
                    run_id=None,  # 유지보수에서는 별도 run_id 없음
                )

                status = result.get("status")
                last_ts = result.get("last_filled_ts")

                if status == "COMPLETE":
                    upsert_rest_progress(run_id, sym, interval, "SUCCESS", last_ts)
                    success_cnt += 1
                else:
                    upsert_rest_progress(run_id, sym, interval, "FAILURE", None)
            except Exception as e:
                logger.exception(f"[REST] 유지보수 오류: {sym} {interval}: {e}")
                upsert_rest_progress(run_id, sym, interval, "FAILURE", None, str(e))

    # ----------------------------------
    # 5) 전체 작업 요약
    # ----------------------------------
    logger.info(f"[REST] total={total_jobs}, success={success_cnt}")

    return {
        "status": "SUCCESS" if success_cnt == total_jobs else "PARTIAL",
        "run_id": run_id,
    }
