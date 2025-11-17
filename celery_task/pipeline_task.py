# celery_task/pipeline_task.py
from __future__ import annotations

import time
import uuid

import httpx
from celery import group
from loguru import logger
from sqlalchemy.dialects.postgresql import insert

from . import celery_app
from . import websocket_task
from .rest_api_task import backfill_symbol_interval

from db_module.connect_sqlalchemy_engine import SyncSessionLocal
from models import CryptoInfo, OHLCV_MODELS
from models.pipeline_state import (
    is_pipeline_active,
    set_component_active,
    PipelineComponent,
)
from models.backfill_progress import BackfillProgress


@celery_app.task(name="pipeline.start_pipeline")
def start_pipeline():
    """
    전체 데이터 수집 파이프라인 구동 태스크
    """
    logger.info("[pipeline.start_pipeline] 파이프라인 시작")

    # 파이프라인 OFF면 즉시 종료
    if not is_pipeline_active():
        logger.info("[pipeline.start_pipeline] pipeline_state.id=1 이 OFF라서 종료")
        return

    # 1) WebSocket 엔진 시작
    set_component_active(PipelineComponent.WEBSOCKET, True)
    websocket_task.websocket_collector.delay()
    logger.info("[pipeline.start_pipeline] WebSocket collector started.")

    # 2) Backfill 시작 전 30초 대기
    time.sleep(30)

    if not is_pipeline_active():
        logger.info("[pipeline.start_pipeline] 중간 OFF 감지 → Backfill 생략")
        set_component_active(PipelineComponent.WEBSOCKET, False)
        return

    # 3) Binance serverTime 조회 → ws_frontier_ms 설정
    try:
        with httpx.Client(timeout=10.0) as client:
            res = client.get("https://fapi.binance.com/fapi/v1/time")
            res.raise_for_status()
            server_time_ms = res.json()["serverTime"]
    except Exception as e:
        logger.error(f"[pipeline.start_pipeline] serverTime 조회 실패: {e}")
        set_component_active(PipelineComponent.BACKFILL, False)
        return

    ws_frontier_ms = int(server_time_ms)
    logger.info(f"[pipeline.start_pipeline] ws_frontier_ms={ws_frontier_ms}")

    if not is_pipeline_active():
        logger.info("[pipeline.start_pipeline] pipeline OFF 감지 → Backfill 생략")
        set_component_active(PipelineComponent.WEBSOCKET, False)
        return

    # 4) Backfill 엔진 ON
    set_component_active(PipelineComponent.BACKFILL, True)

    # ➤ Backfill 전체 작업을 대표하는 run_id 생성
    backfill_run_id = f"pipeline-{uuid.uuid4().hex}"
    logger.info(f"[pipeline.start_pipeline] Backfill run_id={backfill_run_id}")

    # 5) DB에서 심볼과 interval 목록 조회
    with SyncSessionLocal() as session:
        symbols = (
            session.query(CryptoInfo.symbol, CryptoInfo.pair)
            .filter(CryptoInfo.pair.isnot(None))
            .all()
        )

    intervals = list(OHLCV_MODELS.keys())

    if not symbols or not intervals:
        logger.error(
            "[pipeline.start_pipeline] 심볼 또는 interval 목록이 비어 있어 Backfill 불가"
        )
        set_component_active(PipelineComponent.BACKFILL, False)
        return

    # ⚠️ dummy row에 사용될 실제 존재하는 값(첫 번째 심볼, 첫 번째 interval)
    first_symbol = symbols[0].symbol
    first_interval = intervals[0]

    # ⚠️ dummy row 입력 (프론트엔드에서 run_id를 먼저 감지하기 위함)
    with SyncSessionLocal() as session, session.begin():
        stmt = (
            insert(BackfillProgress)
            .values(
                run_id=backfill_run_id,
                symbol=first_symbol,  # ✔ FK 존재
                interval=first_interval,  # ✔ FK 존재
                state="PENDING",
                pct_time=0.0,
                last_candle_ts=None,
                last_error=None,
            )
            .on_conflict_do_nothing()
        )
        session.execute(stmt)

    logger.info(
        f"[pipeline.start_pipeline] Backfill dummy row inserted with symbol={first_symbol}, interval={first_interval}"
    )

    # 6) Backfill 태스크 생성
    jobs = []
    for row in symbols:
        sym = row.symbol
        pair = row.pair

        for interval in intervals:
            jobs.append(
                backfill_symbol_interval.s(
                    symbol=sym,
                    pair=pair,
                    interval=interval,
                    ws_frontier_ms=ws_frontier_ms,
                    run_id=backfill_run_id,  # ➤ 반드시 포함
                )
            )

    if not jobs:
        logger.warning("[pipeline.start_pipeline] Backfill 대상 없음.")
        set_component_active(PipelineComponent.BACKFILL, False)
        return

    # 7) Celery Group 실행
    group_result = group(jobs).apply_async()
    logger.info(
        f"[pipeline.start_pipeline] Backfill group started. total_tasks={len(jobs)}"
    )

    # 8) Backfill 완료될 때까지 Polling
    while not group_result.ready():
        if not is_pipeline_active():
            logger.info("[pipeline.start_pipeline] pipeline OFF → Backfill revoke")
            try:
                group_result.revoke(terminate=True)
            except Exception as e:
                logger.exception(f"Backfill 그룹 revoke 중 오류: {e}")
            break
        time.sleep(5)

    # 9) Backfill 엔진 비활성화
    set_component_active(PipelineComponent.BACKFILL, False)
    logger.info("[pipeline.start_pipeline] Backfill group finished or cancelled.")


@celery_app.task(name="pipeline.stop_pipeline")
def stop_pipeline():
    logger.info("[pipeline.stop_pipeline] 파이프라인 정지 태스크 실행.")
    return
