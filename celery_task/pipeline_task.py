# celery_task/pipeline_task.py
from __future__ import annotations

import time
import uuid

import httpx
from celery import group
from loguru import logger
from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert

from . import celery_app
from . import websocket_task
from .rest_api_task import backfill_symbol_interval

from db_module.connect_sqlalchemy_engine import SyncSessionLocal
from models import CryptoInfo
from models.pipeline_state import (
    is_pipeline_active,
    set_component_active,
    PipelineComponent,
)
from models.backfill_progress import BackfillProgress
from celery_task.rest_maintenance_task import run_rest_maintenance

from celery_task.indicator_task import update_last_indicator_for_symbol_interval

__all__ = ["start_pipeline", "stop_pipeline", "run_maintenance_cycle"]


# ================================================================
# Backfill 전체 완료 여부 판단
# ================================================================
def is_backfill_done(run_id: str) -> bool:
    with SyncSessionLocal() as session:
        rows = (
            session.execute(
                select(BackfillProgress.state).where(BackfillProgress.run_id == run_id)
            )
            .scalars()
            .all()
        )

    if not rows:
        return False

    # 1개라도 FAILURE → 종료 불가
    if any(state == "FAILURE" for state in rows):
        return False

    # 모두 SUCCESS일 때만 OK
    return all(state == "SUCCESS" for state in rows)


# ================================================================
# 1) 전체 파이프라인 시작 (WebSocket → Backfill → Maintenance)
# ================================================================
@celery_app.task(name="pipeline.start_pipeline")
def start_pipeline():
    logger.info("[pipeline] 파이프라인 시작")

    if not is_pipeline_active():
        logger.info("[pipeline] pipeline_state.id=1 이 OFF → 종료")
        return

    # -----------------------------
    # 1) WebSocket 엔진 시작
    # -----------------------------
    set_component_active(PipelineComponent.WEBSOCKET, True)
    websocket_task.websocket_collector.delay()
    logger.info("[pipeline] WebSocket collector started")

    # 안정화를 위해 30초 대기
    time.sleep(30)

    if not is_pipeline_active():
        set_component_active(PipelineComponent.WEBSOCKET, False)
        return

    # -----------------------------
    # 2) Binance 서버 시간 조회
    # -----------------------------
    try:
        with httpx.Client(timeout=10.0) as client:
            res = client.get("https://fapi.binance.com/fapi/v1/time")
            res.raise_for_status()
            server_time_ms = int(res.json()["serverTime"])
    except Exception as e:
        logger.error(f"[pipeline] serverTime 조회 실패: {e}")
        return

    ws_frontier_ms = server_time_ms
    logger.info(f"[pipeline] ws_frontier_ms={ws_frontier_ms}")

    # -----------------------------
    # 3) Backfill 시작
    # -----------------------------
    set_component_active(PipelineComponent.BACKFILL, True)
    run_id = f"run-{uuid.uuid4().hex}"

    logger.info(f"[pipeline] Backfill run_id={run_id}")

    # -----------------------------
    # 심볼 가져오기
    # -----------------------------
    with SyncSessionLocal() as session:
        symbols = (
            session.query(CryptoInfo.symbol, CryptoInfo.pair)
            .filter(CryptoInfo.pair.isnot(None))
            .all()
        )

    intervals = ["1h", "4h", "1d", "1w", "1M"]

    # -----------------------------
    # BackfillProgress Dummy row 생성 (모든 symbol×interval)
    # -----------------------------
    with SyncSessionLocal() as session, session.begin():
        for sym, _pair in symbols:
            for interval in intervals:
                stmt = (
                    insert(BackfillProgress)
                    .values(
                        run_id=run_id,
                        symbol=sym,
                        interval=interval,
                        state="PENDING",
                        pct_time=0.0,
                        last_candle_ts=None,
                        last_error=None,
                    )
                    .on_conflict_do_nothing()
                )
                session.execute(stmt)

    logger.info("[pipeline] BackfillProgress dummy rows inserted")

    # -----------------------------
    # Backfill 병렬 잡 생성
    # -----------------------------
    jobs = []
    for sym, pair in symbols:
        for interval in intervals:
            jobs.append(
                backfill_symbol_interval.s(
                    symbol=sym,
                    pair=pair,
                    interval=interval,
                    ws_frontier_ms=ws_frontier_ms,
                    run_id=run_id,
                )
            )

    g = group(jobs).apply_async()

    logger.info("[pipeline] Backfill group started")

    # -----------------------------
    # Backfill 완료될 때까지 polling
    # -----------------------------
    while is_pipeline_active():
        if g.ready() and is_backfill_done(run_id):
            break
        time.sleep(2)

    set_component_active(PipelineComponent.BACKFILL, False)
    logger.info("[pipeline] Backfill 완료")

    # -----------------------------
    # Backfill 실패 감지
    # -----------------------------
    if not is_backfill_done(run_id):
        logger.error("[pipeline] Backfill 실패 → Maintenance 진입 중단")
        return

    # -----------------------------
    # Backfill 전체 성공 → Maintenance로 이동
    # -----------------------------
    logger.info("[pipeline] Backfill SUCCESS → Maintenance 사이클 시작")
    run_maintenance_cycle.delay()


# ================================================================
# 2) 파이프라인 OFF
# ================================================================
@celery_app.task(name="pipeline.stop_pipeline")
def stop_pipeline():
    logger.info("[pipeline] 전체 pipeline OFF")
    return


# ================================================================
# 3) Backfill 종료 이후 → REST ↔ Indicator 반복
# ================================================================
@celery_app.task(name="pipeline.run_maintenance_cycle")
def run_maintenance_cycle():

    logger.info("[pipeline] Maintenance cycle started")

    while is_pipeline_active():

        # 🔵 REST 유지보수
        set_component_active(PipelineComponent.REST_MAINTENANCE, True)
        logger.info("[pipeline] REST 유지보수 시작")

        rest_job = run_rest_maintenance.delay()
        while not rest_job.ready():
            if not is_pipeline_active():
                return
            time.sleep(1)

        set_component_active(PipelineComponent.REST_MAINTENANCE, False)
        logger.info("[pipeline] REST 유지보수 종료")

        # 🟡 Indicator
        set_component_active(PipelineComponent.INDICATOR, True)
        logger.info("[pipeline] Indicator 계산 시작")

        ind_job = update_last_indicator_for_symbol_interval.delay()
        while not ind_job.ready():
            if not is_pipeline_active():
                return
            time.sleep(1)

        set_component_active(PipelineComponent.INDICATOR, False)
        logger.info("[pipeline] Indicator 계산 완료")

        time.sleep(1)

    logger.info("[pipeline] Maintenance loop stopped")
