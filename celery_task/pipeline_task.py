# celery_task/pipeline_task.py
from __future__ import annotations

import time
import uuid

import httpx
from celery import group
from loguru import logger
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import select

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

__all__ = ["start_pipeline", "stop_pipeline", "run_maintenance_cycle"]


# ================================================================
#  1) 전체 파이프라인 시작
# ================================================================
@celery_app.task(name="pipeline.start_pipeline")
def start_pipeline():
    logger.info("[pipeline.start_pipeline] 파이프라인 시작")

    if not is_pipeline_active():
        logger.info("[pipeline.start_pipeline] pipeline_state.id=1 이 OFF라서 종료")
        return

    # -----------------------------
    # 1) WebSocket 엔진 시작
    # -----------------------------
    set_component_active(PipelineComponent.WEBSOCKET, True)
    websocket_task.websocket_collector.delay()
    logger.info("[pipeline.start_pipeline] WebSocket collector started.")

    # WebSocket 안정화를 위해 30초 대기
    time.sleep(30)

    if not is_pipeline_active():
        logger.info("[pipeline.start_pipeline] OFF 감지 → Backfill 생략")
        set_component_active(PipelineComponent.WEBSOCKET, False)
        return

    # -----------------------------
    # 2) Binance serverTime 조회
    # -----------------------------
    try:
        with httpx.Client(timeout=10.0) as client:
            res = client.get("https://fapi.binance.com/fapi/v1/time")
            res.raise_for_status()
            server_time_ms = res.json()["serverTime"]
    except Exception as e:
        logger.error(f"[pipeline.start_pipeline] serverTime 조회 실패: {e}")
        return

    ws_frontier_ms = int(server_time_ms)
    logger.info(f"[pipeline.start_pipeline] ws_frontier_ms={ws_frontier_ms}")

    # -----------------------------
    # 3) Backfill 시작
    # -----------------------------
    set_component_active(PipelineComponent.BACKFILL, True)

    backfill_run_id = f"pipeline-{uuid.uuid4().hex}"
    logger.info(f"[pipeline.start_pipeline] Backfill run_id={backfill_run_id}")

    # -----------------------------
    # DB에서 심볼 가져오기
    # -----------------------------
    with SyncSessionLocal() as session:
        symbols = (
            session.query(CryptoInfo.symbol, CryptoInfo.pair)
            .filter(CryptoInfo.pair.isnot(None))
            .all()
        )

    # ⚠ 테스트용 interval 강제 지정
    intervals = ["1h", "4h", "1d", "1w", "1M"]

    if not symbols:
        logger.error("[pipeline.start_pipeline] 심볼 없음 → 종료")
        return

    # -----------------------------
    # Dummy row 생성 (모든 interval)
    # -----------------------------
    first_symbol = symbols[0].symbol
    with SyncSessionLocal() as session, session.begin():
        for interval in intervals:
            stmt = (
                insert(BackfillProgress)
                .values(
                    run_id=backfill_run_id,
                    symbol=first_symbol,
                    interval=interval,
                    state="PENDING",
                    pct_time=0.0,
                    last_candle_ts=None,
                    last_error=None,
                )
                .on_conflict_do_nothing()
            )
            session.execute(stmt)

    logger.info("[pipeline.start_pipeline] Dummy rows inserted for all intervals")

    # -----------------------------
    # Backfill 전체 작업 생성
    # -----------------------------
    jobs = []
    for row in symbols:
        for interval in intervals:
            jobs.append(
                backfill_symbol_interval.s(
                    symbol=row.symbol,
                    pair=row.pair,
                    interval=interval,
                    ws_frontier_ms=ws_frontier_ms,
                    run_id=backfill_run_id,
                )
            )

    if not jobs:
        logger.warning("[pipeline.start_pipeline] Backfill job 없음")
        return

    # -----------------------------
    # Backfill 그룹 실행
    # -----------------------------
    group_result = group(jobs).apply_async()
    logger.info("[pipeline.start_pipeline] Backfill group started")

    # 완료될 때까지 Polling
    while not group_result.ready():
        if not is_pipeline_active():
            logger.info("[pipeline.start_pipeline] OFF 감지 → Backfill 중단")
            try:
                group_result.revoke(terminate=True)
            except Exception:
                pass
            break
        time.sleep(3)

    set_component_active(PipelineComponent.BACKFILL, False)
    logger.info("[pipeline.start_pipeline] Backfill group finished")

    # -----------------------------
    # Backfill 전체 성공 여부 확인
    # -----------------------------
    with SyncSessionLocal() as session:
        failures = (
            session.execute(
                select(BackfillProgress).where(
                    BackfillProgress.run_id == backfill_run_id,
                    BackfillProgress.state == "FAILURE",
                )
            )
            .scalars()
            .all()
        )

    if failures:
        logger.error(
            "[pipeline.start_pipeline] Backfill 실패 발생 → Maintenance 진입 차단"
        )
        return

    logger.info("[pipeline.start_pipeline] Backfill 전체 SUCCESS → Maintenance로 이동")

    # -----------------------------------------
    # 4) 유지보수 루프 시작
    # -----------------------------------------
    run_maintenance_cycle.delay()
    logger.info("[pipeline.start_pipeline] Maintenance cycle started.")


# ================================================================
#  2) 파이프라인 정지
# ================================================================
@celery_app.task(name="pipeline.stop_pipeline")
def stop_pipeline():
    logger.info("[pipeline.stop_pipeline] 파이프라인 정지")
    return


# ================================================================
#  3) Backfill 후 REST ↔ Indicator 무한 반복 루프
# ================================================================
@celery_app.task(name="pipeline.run_maintenance_cycle")
def run_maintenance_cycle():
    logger.info("[pipeline] Maintenance cycle started")

    from .rest_maintenance_task import run_maintenance_cycle as rest_cycle
    from .indicator_task import update_last_indicator_for_symbol_interval as ind_cycle

    while is_pipeline_active():

        # 🔵 REST 유지보수
        logger.info("[pipeline] REST maintenance 시작")
        set_component_active(PipelineComponent.REST_MAINTENANCE, True)

        rest_job = rest_cycle.delay()
        while not rest_job.ready():
            if not is_pipeline_active():
                set_component_active(PipelineComponent.REST_MAINTENANCE, False)
                return
            time.sleep(2)

        set_component_active(PipelineComponent.REST_MAINTENANCE, False)
        logger.info("[pipeline] REST maintenance 완료")

        # 🟡 보조지표 계산
        logger.info("[pipeline] Indicator 계산 시작")
        set_component_active(PipelineComponent.INDICATOR, True)

        ind_job = ind_cycle.delay()
        while not ind_job.ready():
            if not is_pipeline_active():
                set_component_active(PipelineComponent.INDICATOR, False)
                return
            time.sleep(2)

        set_component_active(PipelineComponent.INDICATOR, False)
        logger.info("[pipeline] Indicator 계산 완료")

        time.sleep(1)

    logger.info("[pipeline] Maintenance cycle stopped (pipeline OFF)")
